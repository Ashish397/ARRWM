"""GAN G/D BALANCE tripwires.

A separate module, deliberately: it has no dependency on the disc, the
trainer or any feature source, so it can be unit-tested against a replayed
``d_loss`` trajectory from a real run log and cannot be broken by the
three agents currently editing ``model/ladd_disc.py`` /
``model/ladd_pixel_features.py``.

WHY THIS EXISTS
===============
2026-08-26. The arm ``pixdirect_strong`` (encoder ``lr_scale`` 0.4 +
``ladd_pixel_crops_per_row`` 1->2 + ``ladd_pixel_lat_frames`` 2->3) drove
the discriminator into the documented **D-WINS COLLAPSE**. Its logged
``d_loss`` trajectory (``ln2 = 0.6931`` is chance; the healthy band is
0.25-0.55; the documented failure threshold is ``d_loss <= 0.135``
sustained, from the ``w003`` verdict):

    31 .6685  41 .3585  51 .6312  61 .4138  71 .4322  81 .6700  91 .6644
   101 .3632 111 .0431 121 .0482 131 .4134 141 .4907 151 .3268
   161 .0104 171 .0039 181 .0024 191 .0102     <- FOUR consecutive, still
                                                  falling at run end

The **median over steps >= 50 is 0.3632**, which reads as a healthy-band
success. It is an artefact of averaging a healthy first half against a
collapsed second half: the campaign was one report away from calling this
the best arm. The contrasting arm ``pixdirect_onlinelr`` (same encoder
lr, coverage NOT raised) dips to 0.1346 / 0.0139 at steps 111/121 and
then **recovers** to 0.6243 -- a transient, not a collapse. A median
cannot tell those two apart. A consecutive-run counter can.

WHAT IT IS AND IS NOT
=====================
* It is **observability only**. It never touches a tensor, an optimiser,
  the RNG or the loss, so a run with the tripwire enabled trains
  byte-identically to one without it. It reports; it does not intervene.
* It is a **latch**: once tripped it stays tripped and records the step,
  because the question at read time is "did this run ever collapse", not
  "is it collapsed right now".
* Exactly ``0.0`` is **rejected, not counted**. Before
  ``gan_disc_start_step`` the GAN path reports ``d_loss=0.0000`` as a
  not-run sentinel (see steps 11 and 21 of both arms above). A tripwire
  that counted those would fire on every run at step 21 and would be
  worth nothing -- the forgeable-zero failure mode, in the one place
  where it would be most damaging.
"""
from typing import Dict, Optional

__all__ = ["DWinsTripwire", "DEFAULT_DWINS_FLOOR", "DEFAULT_DWINS_K"]

# The documented D-wins threshold. Source: the ``w003`` verdict --
# "d_loss <= 0.135 for five consecutive logged steps" against a
# 0.25-0.55 healthy band -- recorded in docs/ONBOARDING_PIXDIRECT.md §6.
DEFAULT_DWINS_FLOOR = 0.135
# 3, not 5: ``pixdirect_onlinelr``'s recovered transient is 2 consecutive
# observations, so 3 is the smallest K that separates the measured
# recovery from the measured collapse. Configurable.
DEFAULT_DWINS_K = 3


class DWinsTripwire:
    """Consecutive-``d_loss``-below-floor detector.

    Feed it one ``d_loss`` per logged step; read the counters it returns.

    Args:
        floor: ``d_loss`` at or below this counts as "the disc is
            winning". Default ``DEFAULT_DWINS_FLOOR`` (0.135).
        k_consecutive: how many consecutive observations at or below
            ``floor`` trip the latch. Default ``DEFAULT_DWINS_K`` (3).
        min_step: observations before this step are ignored entirely
            (not counted, not streak-breaking). Set it to
            ``gan_disc_start_step``; before that the disc has not run.
    """

    def __init__(
        self,
        floor: float = DEFAULT_DWINS_FLOOR,
        k_consecutive: int = DEFAULT_DWINS_K,
        min_step: int = 0,
    ) -> None:
        self.floor = float(floor)
        self.k = max(1, int(k_consecutive))
        self.min_step = int(min_step)
        self.streak = 0
        self.max_streak = 0
        self.below_total = 0
        self.observations = 0
        self.skipped_sentinel = 0
        self.tripped = False
        self.trip_step = -1
        self.last_d_loss = float("nan")

    # ------------------------------------------------------------------
    def observe(self, step: int, d_loss: Optional[float]) -> bool:
        """Record one logged ``d_loss``. Returns True on the trip EDGE
        (the single observation that latches it), False otherwise -- so
        the caller can emit exactly one loud warning per run.

        Rejected without counting, each for a stated reason:
          * ``d_loss is None``          -- the field was absent
          * ``d_loss != d_loss``        -- NaN
          * ``step < min_step``         -- the disc had not started
          * ``d_loss == 0.0`` exactly   -- the not-run sentinel. An RpGAN
            softplus mean is never exactly zero, so this cannot discard a
            real collapse; it discards only the pre-start rows.
        """
        if d_loss is None:
            return False
        v = float(d_loss)
        if v != v:
            return False
        if int(step) < self.min_step:
            return False
        if v == 0.0:
            self.skipped_sentinel += 1
            return False
        self.observations += 1
        self.last_d_loss = v
        if v <= self.floor:
            self.streak += 1
            self.below_total += 1
            self.max_streak = max(self.max_streak, self.streak)
            if self.streak >= self.k and not self.tripped:
                self.tripped = True
                self.trip_step = int(step)
                return True
        else:
            self.streak = 0
        return False

    # ------------------------------------------------------------------
    def message(self) -> str:
        return (
            "[GAN-BALANCE] D-WINS TRIPWIRE: d_loss <= %.3f for %d "
            "consecutive logged steps (last=%.4g, step=%d). The "
            "discriminator has separated real from fake essentially "
            "perfectly, so the generator is receiving no useful "
            "adversarial gradient -- every texture result from this "
            "point on is void. A run-level MEDIAN will NOT show this: "
            "pixdirect_strong's median over steps>=50 was 0.3632, inside "
            "the healthy band, while its last four logged steps were "
            "0.0104/0.0039/0.0024/0.0102. Do NOT fix it with "
            "gan_loss_weight (it scales only the G side and converts "
            "over-driven-G into exactly this failure); move gan_lr or "
            "gan_updates_per_step, which scale BOTH sides."
            % (self.floor, self.streak, self.last_d_loss, self.trip_step)
        )

    # ------------------------------------------------------------------
    def logs(self, prefix: str = "train/gan_dwins_") -> Dict[str, float]:
        """Telemetry. ``_tripped`` is the headline; ``_max_streak`` is
        the one to read on a run that did not trip, because it says how
        close it came."""
        return {
            prefix + "tripped": 1.0 if self.tripped else 0.0,
            prefix + "trip_step": float(self.trip_step),
            prefix + "streak": float(self.streak),
            prefix + "max_streak": float(self.max_streak),
            prefix + "below_floor_total": float(self.below_total),
            prefix + "observations": float(self.observations),
            prefix + "skipped_sentinel": float(self.skipped_sentinel),
            prefix + "floor": float(self.floor),
            prefix + "k": float(self.k),
        }
