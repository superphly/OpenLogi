//! Lazy per-device load state for background HID++ reads, shared by DPI,
//! SmartShift, and onboard-profile discovery.

use std::collections::{BTreeMap, BTreeSet};

use openlogi_core::hid::{DpiInfo, OnboardProfilesInfo, SmartShiftStatus, WriteError};
use tracing::debug;

use super::device_key::DeviceKey;

/// How many times to retry a device read after a transient HID++ error
/// (read timeout, busy device)
/// before giving up. A genuine "feature not supported" reply is permanent and
/// never retried.
const LOAD_MAX_ATTEMPTS: u8 = 3;

/// Lazy per-device load state for a background HID++ read: unqueried, in flight,
/// resolved, transiently failed (retryable on re-select), or permanently
/// unsupported. Shared by DPI capability discovery and SmartShift reads through
/// [`LazyDeviceData`]; the two differ only in payload type `T` and in which
/// errors count as permanent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Load<T> {
    /// The selected device has not been queried yet.
    Unknown,
    /// A background HID++ read is in flight.
    Loading,
    /// The device reported its value.
    Ready(T),
    /// Transient errors (read timeouts, busy device) exhausted the retry budget.
    /// Distinct from [`Self::Unsupported`] because the device may well support
    /// the feature — re-selecting it (see
    /// [`AppState::set_current_device`](super::AppState::set_current_device))
    /// grants a fresh attempt.
    Failed(String),
    /// The device genuinely does not support the feature; never retried.
    Unsupported(String),
}

/// Per-device DPI capability load state. See [`Load`].
pub type DpiStatus = Load<DpiInfo>;

/// Per-device SmartShift (`0x2111`) config load state. See [`Load`]. Unlike DPI
/// presets, the resolved config is *not* persisted to `config.toml` — the device
/// stores wheel mode / threshold / torque in its own non-volatile memory, so the
/// GUI only ever reads and writes the device.
pub type SmartShiftLoad = Load<SmartShiftStatus>;

/// Per-device onboard-profile (`0x8100`) state load. See [`Load`].
pub type ProfilesLoad = Load<OnboardProfilesInfo>;

/// The lazily-loaded DPI and SmartShift read caches, grouped so callers reach
/// them as `state.reads.dpi` / `state.reads.smartshift` and use
/// [`LazyDeviceData`]'s own methods directly — instead of `AppState` growing
/// a same-shaped forwarding method twice, once per subsystem, for every
/// operation the generic already provides.
#[derive(Default)]
pub(crate) struct DeviceReads {
    pub(crate) dpi: LazyDeviceData<DpiInfo>,
    pub(crate) smartshift: LazyDeviceData<SmartShiftStatus>,
    pub(crate) profiles: LazyDeviceData<OnboardProfilesInfo>,
}

/// Per-device lazy-load cache for a background HID++ read, keyed by
/// [`DeviceKey`]. Holds each device's [`Load`] state plus its transient-retry
/// counter, and carries the stale-route guard + retry-budget policy once, for
/// both DPI and SmartShift.
pub(crate) struct LazyDeviceData<T> {
    by_device: BTreeMap<DeviceKey, Load<T>>,
    /// Consecutive transient read failures per device, capped by
    /// [`LOAD_MAX_ATTEMPTS`] before the device settles on [`Load::Failed`].
    attempts: BTreeMap<DeviceKey, u8>,
    /// [`Load::Ready`] entries whose payload may no longer match the device
    /// (the value can change on-device: DPI button, onboard profile switch).
    /// A stale entry keeps rendering its cached payload while the next render
    /// issues a silent re-read — see [`Self::mark_stale`] /
    /// [`Self::begin_refresh`].
    stale: BTreeSet<DeviceKey>,
    /// Devices with a silent refresh in flight, so a render loop doesn't
    /// re-issue one per frame.
    refreshing: BTreeSet<DeviceKey>,
}

// Manual `Default` (not derived): a derive would demand `T: Default`, but the
// empty maps need nothing of `T`.
impl<T> Default for LazyDeviceData<T> {
    fn default() -> Self {
        Self {
            by_device: BTreeMap::new(),
            attempts: BTreeMap::new(),
            stale: BTreeSet::new(),
            refreshing: BTreeSet::new(),
        }
    }
}

impl<T: Clone> LazyDeviceData<T> {
    /// The recorded state for `key`, or [`Load::Unknown`] if never queried.
    pub(crate) fn status(&self, key: &DeviceKey) -> Load<T> {
        self.by_device.get(key).cloned().unwrap_or(Load::Unknown)
    }

    /// The raw recorded entry for `key`, for callers that match on `Ready`
    /// without cloning the payload.
    pub(crate) fn get(&self, key: &DeviceKey) -> Option<&Load<T>> {
        self.by_device.get(key)
    }

    /// Whether `key` still needs a read (nothing recorded yet). Cheaper than
    /// cloning [`status`](Self::status) on the per-frame render path.
    pub(crate) fn unqueried(&self, key: &DeviceKey) -> bool {
        !self.by_device.contains_key(key)
    }

    /// Mark a read as in flight for `key`.
    pub(crate) fn mark_loading(&mut self, key: &DeviceKey) {
        self.by_device.insert(key.clone(), Load::Loading);
    }

    /// Reset a stuck `Loading` for `key` back to unqueried — the read worker
    /// vanished (e.g. panicked) without delivering a result, so the next render
    /// re-issues instead of wedging the device on "Reading…".
    pub(crate) fn clear_loading(&mut self, key: &DeviceKey) {
        if matches!(self.by_device.get(key), Some(Load::Loading)) {
            self.by_device.remove(key);
        }
    }

    /// Drop `key`'s recorded state and retry budget so the next render re-reads.
    /// Backs the "click to retry" affordance and the re-select-grants-a-retry
    /// rule for a [`Load::Failed`] device.
    pub(crate) fn retry(&mut self, key: &DeviceKey) {
        self.by_device.remove(key);
        self.attempts.remove(key);
        self.stale.remove(key);
        self.refreshing.remove(key);
    }

    /// Forget `key` entirely — the device disappeared, or reconnected on a new
    /// route, so its cached state (keyed to the dead route) is stale.
    pub(crate) fn remove(&mut self, key: &DeviceKey) {
        self.by_device.remove(key);
        self.attempts.remove(key);
        self.stale.remove(key);
        self.refreshing.remove(key);
    }

    /// Forget every device the `present` predicate rejects (not in the live set).
    pub(crate) fn retain_present(&mut self, present: impl Fn(&str) -> bool) {
        self.by_device.retain(|key, _| present(key.as_str()));
        self.attempts.retain(|key, _| present(key.as_str()));
        self.stale.retain(|key| present(key.as_str()));
        self.refreshing.retain(|key| present(key.as_str()));
    }

    /// Flag `key`'s resolved value as possibly out of date — the device can
    /// change it on its own (DPI button press, onboard profile activation).
    /// The cached payload keeps rendering; the next render issues a silent
    /// re-read via [`Self::begin_refresh`]. No-op unless `key` is
    /// [`Load::Ready`] with no refresh already in flight, so non-resolved
    /// states keep their initial-load / retry semantics.
    pub(crate) fn mark_stale(&mut self, key: &DeviceKey) {
        if matches!(self.by_device.get(key), Some(Load::Ready(_))) && !self.refreshing.contains(key)
        {
            self.stale.insert(key.clone());
        }
    }

    /// Claim `key`'s stale flag for a refresh read: returns `true` exactly once
    /// per [`Self::mark_stale`], moving the key into the refresh-in-flight set
    /// so a render loop can't issue duplicates. The result lands through
    /// [`Self::store_refresh`], or [`Self::clear_refreshing`] if it never comes.
    pub(crate) fn begin_refresh(&mut self, key: &DeviceKey) -> bool {
        if self.stale.remove(key) {
            self.refreshing.insert(key.clone());
            true
        } else {
            false
        }
    }

    /// Reset a refresh whose reply was dropped, so a later
    /// [`Self::mark_stale`] can try again.
    pub(crate) fn clear_refreshing(&mut self, key: &DeviceKey) {
        self.refreshing.remove(key);
    }

    /// Store a silent-refresh result: a delivered value replaces the cached
    /// one (returned so the caller can seed derived state), while any error
    /// keeps the previous [`Load::Ready`] on screen — a refresh must never
    /// blank a panel that was rendering fine a frame ago.
    pub(crate) fn store_refresh(
        &mut self,
        key: DeviceKey,
        result: Result<T, WriteError>,
        label: &'static str,
    ) -> Option<T> {
        self.refreshing.remove(&key);
        match result {
            Ok(value) => {
                self.by_device.insert(key, Load::Ready(value.clone()));
                Some(value)
            }
            Err(error) => {
                debug!(key = %key, error = %error, label, "silent refresh failed — keeping cached value");
                None
            }
        }
    }

    /// Optimistically record a resolved value with no read involved — e.g. a
    /// just-written SmartShift config, shown until a confirming re-read replaces
    /// it. Leaves the retry budget untouched.
    pub(crate) fn set_ready(&mut self, key: DeviceKey, value: T) {
        self.by_device.insert(key, Load::Ready(value));
    }

    /// Store a read result under the stale-route guard and the transient-retry /
    /// permanent-unsupported policy. `matches_route` is whether a live device
    /// still holds `key` *on the route the read targeted*; `still_present` is
    /// whether `key` exists at all. Returns the resolved value when the result
    /// settled to [`Load::Ready`], so the caller can run a side effect (the DPI
    /// panel seeds the shared current value). `label` tags the debug logs.
    pub(crate) fn store(
        &mut self,
        key: DeviceKey,
        result: Result<T, WriteError>,
        is_permanent: impl Fn(&WriteError) -> bool,
        matches_route: bool,
        still_present: bool,
        label: &'static str,
    ) -> Option<T> {
        if !matches_route {
            debug!(key = %key, label, "stale device read result ignored");
            // The device reconnected on a different route mid-read: drop the
            // orphaned `Loading` marker so the next render re-reads against the
            // live route instead of spinning on "Reading…" forever.
            if still_present {
                self.by_device.remove(&key);
            }
            return None;
        }

        let status = match result {
            Ok(value) => {
                self.attempts.remove(&key);
                Load::Ready(value)
            }
            // A genuine "feature not supported" reply never changes — record it
            // and stop probing.
            Err(error) if is_permanent(&error) => {
                self.attempts.remove(&key);
                Load::Unsupported(error.to_string())
            }
            // Transient failures get a few more tries: clear the status so the
            // next render re-reads, until the budget runs out, then settle on
            // `Failed` (retryable on re-select) rather than `Unsupported`.
            Err(error) => {
                let attempts = self.attempts.entry(key.clone()).or_insert(0);
                *attempts = attempts.saturating_add(1);
                if *attempts < LOAD_MAX_ATTEMPTS {
                    debug!(key = %key, attempts = *attempts, error = %error, label, "transient device read error — will retry");
                    self.by_device.remove(&key);
                    return None;
                }
                self.attempts.remove(&key);
                Load::Failed(error.to_string())
            }
        };

        // Clone out the resolved value (cheap; once per completed read) before
        // the status moves into the map, so the caller can seed derived state
        // without re-borrowing `self`.
        let resolved = match &status {
            Load::Ready(value) => Some(value.clone()),
            _ => None,
        };
        self.by_device.insert(key, status);
        resolved
    }
}

#[cfg(test)]
mod tests {
    use openlogi_core::hid::WriteError;

    use super::{DeviceKey, LazyDeviceData, Load};

    fn key() -> DeviceKey {
        DeviceKey::from("receiver:test:slot:1")
    }

    #[test]
    fn mark_stale_only_flags_resolved_entries() {
        let mut data = LazyDeviceData::<u8>::default();
        // Unqueried: nothing to refresh — the initial-load path owns it.
        data.mark_stale(&key());
        assert!(!data.begin_refresh(&key()), "unqueried must not refresh");
        // Loading: same — the in-flight initial read will deliver.
        data.mark_loading(&key());
        data.mark_stale(&key());
        assert!(!data.begin_refresh(&key()), "loading must not refresh");
        // Ready: stale flag arms exactly one refresh.
        data.set_ready(key(), 7);
        data.mark_stale(&key());
        assert!(data.begin_refresh(&key()), "ready+stale must refresh");
        assert!(
            !data.begin_refresh(&key()),
            "the stale flag is claimed once, not once per render frame"
        );
        // While the refresh is in flight, re-marking is a no-op.
        data.mark_stale(&key());
        assert!(
            !data.begin_refresh(&key()),
            "no duplicate in-flight refresh"
        );
    }

    #[test]
    fn store_refresh_keeps_the_cached_value_on_error() {
        let mut data = LazyDeviceData::<u8>::default();
        data.set_ready(key(), 7);
        data.mark_stale(&key());
        assert!(data.begin_refresh(&key()), "refresh should arm");
        let stored = data.store_refresh(key(), Err(WriteError::DeviceNotFound), "test");
        assert_eq!(stored, None);
        assert_eq!(
            data.status(&key()),
            Load::Ready(7),
            "a failed silent refresh must not blank the panel"
        );
        // The failure released the in-flight claim: staleness can re-arm.
        data.mark_stale(&key());
        assert!(
            data.begin_refresh(&key()),
            "refresh re-arms after a failure"
        );
        let stored = data.store_refresh(key(), Ok(9), "test");
        assert_eq!(stored, Some(9));
        assert_eq!(data.status(&key()), Load::Ready(9));
    }

    #[test]
    fn remove_clears_the_stale_and_refresh_flags() {
        let mut data = LazyDeviceData::<u8>::default();
        data.set_ready(key(), 7);
        data.mark_stale(&key());
        data.remove(&key());
        assert!(
            !data.begin_refresh(&key()),
            "a removed device leaves no orphaned stale flag"
        );
    }
}
