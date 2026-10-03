import Mathlib

/-!
# Conservation of spend in the online reference-budget history

**Property (stated before looking for defects):** the daily/reset history that the reference budget
reads records every cent the ad set spends, exactly once. Formally, after initialisation, the sum of
`spend` over all history rows equals the increase in true lifetime spend.

Source (fbsource `7a9540fef80d`, `fbcode/admarket/lib/adpacing/pacers/costpacer/`):
* `state/CostPacerStateManager.cpp:626-786` `accumulateNewDelivery` (delta, drop check, watermark)
* `state/CostPacerStateManager.cpp:61-180` `logAccumulatedSnapshots` (rows)
* `state/CostPacerStateManager.cpp:790-845` `updateNewAccumulation` (watermark update / skip)
* `CostPacerPacingContextImpl.cpp:58` `getPacingSpend() = L - W`, `W` re-stamped to `L` on reset.

Money is exact `ℚ` (i.e. **case I already fixed**), so anything found here is a different defect.
Scope: global pacing spend only; `kEpsilon` taken as 0; one advertiser day (day attribution is a
separate property); `bqrtForceReset` folded into `transition`.
-/

namespace Conservation

/-- What the pacer sees in one cycle. -/
structure Obs where
  L : ℚ                 -- `getPacingLifetimeSpend()` as read this cycle
  reset : Bool          -- `DELIVERY_STATS_RESET` signal; `W` is re-stamped to `L` first
  convDrop : Bool       -- conversions / objective value / ranking predictions dropped (`:662-672`)
  transition : Bool     -- input-view transition or algorithm-version change (`:673`)

structure Row where
  spend : ℚ
  isReset : Bool

structure St where
  init : Bool := false  -- `lastAccumulation.adSetPacingTime() > 0`
  W : ℚ := 0            -- `deliveryStatsSnapshot_.lifetimeSpendNativeCents()`
  last : ℚ := 0         -- `spendAtLastAccumulation`
  rows : List Row := []

/-- `logAccumulatedSnapshots`, one day: empty → new row; reset → new row seeded with `P`
(`:110`); otherwise add the delta to the head (`:145`). -/
def logRow (rows : List Row) (delta P : ℚ) (reset : Bool) : List Row :=
  match rows with
  | [] => [⟨delta, reset⟩]
  | r :: rs => if reset then ⟨P, true⟩ :: r :: rs else ⟨r.spend + delta, r.isReset⟩ :: rs

/-- One cycle of `accumulateNewDelivery` + `updateNewAccumulation`. `flag` is
`stateManagementSettings.useExplictStatsResetSignal`. -/
def step (flag : Bool) (s : St) (o : Obs) : St :=
  let W := if o.reset then o.L else s.W
  let P := o.L - W
  if !s.init then { s with init := true, W := W, last := P }   -- `:792-794`: no row, watermark := P
  else
    let drop := decide (s.last - P > 0) || o.convDrop || o.transition          -- `:662-674`
    let delta := if drop then 0 else P - s.last                               -- `:682`
    let falseReset := drop && !o.reset && !o.transition                        -- `:675-676`
    let skip := flag && falseReset                                            -- `:783-784`
    { init := true, W := W,
      last := if skip then s.last else P,                                      -- `:815` / `:845`
      rows := logRow s.rows delta P o.reset }

def run (flag : Bool) (os : List Obs) : St := os.foldl (step flag) {}

def recorded (s : St) : ℚ := (s.rows.map (·.spend)).sum

/-! ## Scenario generator for `plausible`

A scenario is a list of cycles `(inc, dip, reset, convDrop)`: true lifetime spend `T` grows by
`inc` cents; the counter is read `dip` cents low this cycle (a transient stale read, `L = T - dip`). -/

def build (cs : List (ℕ × ℕ × Bool × Bool)) : List Obs × ℚ × ℚ :=
  -- returns the observations, true spend at the first cycle, and at the last cycle
  let rec go (T : ℚ) : List (ℕ × ℕ × Bool × Bool) → List Obs × ℚ
    | [] => ([], T)
    | (inc, dip, r, c) :: rest =>
      let T' := T + inc
      let (os, Tend) := go T' rest
      (⟨max 0 (T' - dip), r, c, false⟩ :: os, Tend)
  match cs with
  | [] => ([], 0, 0)
  | (inc, _, _, _) :: _ =>
    let (os, Tend) := go 0 cs
    (os, (inc : ℚ), Tend)

/-- Conservation: history total = true spend since the first (initialising) cycle. -/
def conserves (flag : Bool) (cs : List (ℕ × ℕ × Bool × Bool)) : Bool :=
  let (os, T0, T1) := build cs
  decide (recorded (run flag os) = T1 - T0)

/-! ## Findings, executed (`recorded` vs true spend since the first cycle)

`o L r c t` = one cycle reading lifetime spend `L`, with reset / conversion-drop / transition flags. -/

def o (L : ℚ) (r c t : Bool := false) : Obs := ⟨L, r, c, t⟩

-- N1. Reset cycle: the spend between the previous cycle and the reset cycle is never recorded
--     (both flag values). True 1, recorded 0.
#eval (recorded (run false [o 0, o 1 (r := true)]), recorded (run true [o 0, o 1 (r := true)]))
-- N2. Conversion/objective counter drops while spend is healthy. True 1.
--     flag off: recorded 0 (lost); flag on: recorded 1 (deferred to the next cycle).
#eval (recorded (run false [o 0, o 1 (c := true), o 1]), recorded (run true [o 0, o 1 (c := true), o 1]))
-- N3. Transient low read of the spend counter (1 → 0 → 1). True 0.
--     flag off: recorded 1 (the dip is counted twice); flag on: recorded 0.
#eval (recorded (run false [o 1, o 0, o 1]), recorded (run true [o 1, o 0, o 1]))
-- N4. Genuine counter decrease (100 → 50), then 20 cents of new spend (→ 70). True new spend 20.
--     flag off: recorded 20; flag on: recorded 0, frozen until the counter passes 100.
#eval (recorded (run false [o 100, o 50, o 60, o 70]), recorded (run true [o 100, o 50, o 60, o 70]))
-- N5. Input-view transition or algorithm change: that cycle's spend is dropped (both flags). True 1.
#eval (recorded (run false [o 0, o 1 (t := true), o 1]), recorded (run true [o 0, o 1 (t := true), o 1]))

end Conservation
