import Mathlib

/-!
# Case I: the online reference-budget history truncates fractional cents

Source (fbsource `7a9540fef80d`, `fbcode/admarket/lib/adpacing/pacers/costpacer/`):

* Branch A, `state/BidFlexingStateManager.cpp:44`: same day, healthy counter, global pacing
  (`dailySpendSplitProratedFromGlobal = 0`): `dailySpend = dailySpend + deltaSpend` in `double`.
* Branch B, `state/CostPacerStateManager.cpp`: the daily history the reference budget reads.
  Field `spend` is `i64` (`if/cost.thrift:258`).
  - `:68`  negative deltas are dropped (`if (deltaSpend < 0 ...) return;`).
  - `:43`  first row of a day: `spend() = inputs.spend` (a `double` delta, converted to `i64`).
  - `:145` same day: `spend() = static_cast<double>(spend) + deltaSpend`, converted to `i64`.
  C++ `double → int64` conversion truncates toward zero.

Scope: one advertiser day, no reset, healthy counter. Money is native cents in `ℚ`, so the only
rounding is the rounding the code does; `double` round-off is out of scope.
-/

namespace CaseI

/-- C++ `double → int64` conversion: truncation toward zero. -/
def toI64 (x : ℚ) : ℤ := if 0 ≤ x then ⌊x⌋ else ⌈x⌉

/-- Branch A, `BidFlexingStateManager.cpp:44`: `dailySpend += deltaSpend` in `double`. -/
def branchA (ds : List ℚ) : ℚ := ds.foldl (· + ·) 0

/-- Branch B as written: the first delta opens the row (`:43`), each later delta is added and the
sum converted back to `i64` (`:145`). -/
def branchB : List ℚ → ℤ
  | [] => 0
  | d :: rest => rest.foldl (fun s e => toI64 ((s : ℚ) + e)) (toI64 d)

/-- The proposed fix: field `spend` declared `double`. -/
def branchBFixed (ds : List ℚ) : ℚ := ds.foldl (· + ·) 0

/-- The other proposed fix: keep `i64`, but persist the fractional remainder. -/
def carryStep (st : ℤ × ℚ) (d : ℚ) : ℤ × ℚ := (st.1 + ⌊st.2 + d⌋, Int.fract (st.2 + d))
def branchBCarry (ds : List ℚ) : ℤ := (ds.foldl carryStep (0, 0)).1

/-! ## Lemmas -/

lemma toI64_of_nonneg {x : ℚ} (h : 0 ≤ x) : toI64 x = ⌊x⌋ := by simp [toI64, h]

lemma foldl_add_start (ds : List ℚ) (a : ℚ) : ds.foldl (· + ·) a = a + ds.sum := by
  induction ds generalizing a with
  | nil => simp
  | cons d ds ih => simp [List.foldl, ih]; ring

lemma branchA_eq_sum (ds : List ℚ) : branchA ds = ds.sum := by
  simp [branchA, foldl_add_start]

/-- One truncating update from a nonnegative integer `s` adds exactly `⌊d⌋`: the stored value is
already an integer, so the fraction lost is the whole fractional part of the delta. -/
lemma trunc_step {s : ℤ} {d : ℚ} (hs : 0 ≤ s) (hd : 0 ≤ d) :
    toI64 ((s : ℚ) + d) = s + ⌊d⌋ := by
  have : (0 : ℚ) ≤ (s : ℚ) + d := by positivity
  rw [toI64_of_nonneg this, Int.floor_intCast_add]

lemma foldl_trunc (ds : List ℚ) (s : ℤ) (hs : 0 ≤ s) (h : ∀ d ∈ ds, 0 ≤ d) :
    ds.foldl (fun s e => toI64 ((s : ℚ) + e)) s = s + (ds.map (fun d => ⌊d⌋)).sum := by
  induction ds generalizing s with
  | nil => simp
  | cons d ds ih =>
    have hd : 0 ≤ d := h d (by simp)
    simp only [List.foldl, List.map, List.sum_cons]
    rw [trunc_step hs hd, ih _ (by have := Int.floor_nonneg.mpr hd; omega)
      (fun e he => h e (by simp [he]))]
    ring

/-! ## Results -/

/-- **Branch B stores exactly `Σ ⌊δᵢ⌋`.** Every update throws away `fract δᵢ`, and nothing carries. -/
theorem branchB_eq_sum_floor (ds : List ℚ) (h : ∀ d ∈ ds, 0 ≤ d) :
    branchB ds = (ds.map (fun d => ⌊d⌋)).sum := by
  cases ds with
  | nil => simp [branchB]
  | cons d rest =>
    have hd : 0 ≤ d := h d (by simp)
    simp only [branchB, List.map, List.sum_cons]
    rw [toI64_of_nonneg hd, foldl_trunc rest _ (Int.floor_nonneg.mpr hd)
      (fun e he => h e (by simp [he]))]

/-- **The loss is exactly `Σ fract δᵢ`.** -/
theorem loss_eq_sum_fract (ds : List ℚ) (h : ∀ d ∈ ds, 0 ≤ d) :
    branchA ds - branchB ds = (ds.map Int.fract).sum := by
  rw [branchA_eq_sum, branchB_eq_sum_floor ds h]
  induction ds with
  | nil => simp
  | cons d ds ih =>
    simp only [List.map, List.sum_cons, Int.cast_add]
    rw [← ih (fun e he => h e (by simp [he]))]
    unfold Int.fract; ring

/-- Branch B never overstates: the bias is one-sided, downward. -/
theorem branchB_le_branchA (ds : List ℚ) (h : ∀ d ∈ ds, 0 ≤ d) : (branchB ds : ℚ) ≤ branchA ds := by
  have := loss_eq_sum_fract ds h
  have : 0 ≤ (ds.map Int.fract).sum :=
    List.sum_nonneg (by simp only [List.mem_map]; rintro _ ⟨d, -, rfl⟩; exact Int.fract_nonneg d)
  linarith

/-- The loss is less than one cent per update (strictly, once there is an update). -/
theorem loss_lt_updates (ds : List ℚ) (h : ∀ d ∈ ds, 0 ≤ d) (hne : ds ≠ []) :
    branchA ds - branchB ds < ds.length := by
  rw [loss_eq_sum_fract ds h]
  induction ds with
  | nil => exact absurd rfl hne
  | cons d ds ih =>
    simp only [List.map, List.sum_cons, List.length_cons, Nat.cast_add, Nat.cast_one]
    have h1 := Int.fract_lt_one d
    cases ds with
    | nil => simp; linarith
    | cons e es =>
      have := ih (fun x hx => h x (by simp [hx])) (List.cons_ne_nil e es); linarith

/-- **Sub-cent updates are lost entirely:** if every delta is below one cent, Branch B stays `0`
however much was spent. This is the `0.999 → 0` pattern in `root_causes.md`. -/
theorem subcent_lost (ds : List ℚ) (h : ∀ d ∈ ds, 0 ≤ d ∧ d < 1) : branchB ds = 0 := by
  rw [branchB_eq_sum_floor ds (fun d hd => (h d hd).1)]
  apply List.sum_eq_zero
  simp only [List.mem_map]
  rintro _ ⟨d, hd, rfl⟩
  exact Int.floor_eq_zero_iff.mpr ⟨(h d hd).1, (h d hd).2⟩

/-- So the relative error is unbounded (100%): for any spend level `M` there is a day whose true
spend exceeds `M` while the stored history is `0`. -/
theorem unbounded_loss (M : ℚ) :
    ∃ ds : List ℚ, (∀ d ∈ ds, 0 ≤ d ∧ d < 1) ∧ M < branchA ds ∧ branchB ds = 0 := by
  obtain ⟨n, hn⟩ := exists_nat_gt (2 * M)
  refine ⟨List.replicate n (1 / 2), ?_, ?_, ?_⟩
  · intro d hd; rw [List.eq_of_mem_replicate hd]; norm_num
  · rw [branchA_eq_sum, List.sum_replicate, nsmul_eq_mul]; linarith
  · exact subcent_lost _ (by intro d hd; rw [List.eq_of_mem_replicate hd]; norm_num)

/-- **Fix 1 (`double` field) restores the invariant `Σ Branch B = Branch A`** of
`online_ref_budget.md` §9. -/
theorem fixed_eq_branchA (ds : List ℚ) : branchBFixed ds = branchA ds := rfl

lemma carry_invariant (ds : List ℚ) (st : ℤ × ℚ) (h0 : 0 ≤ st.2) (h1 : st.2 < 1) :
    let r := ds.foldl carryStep st
    (r.1 : ℚ) + r.2 = st.1 + st.2 + ds.sum ∧ 0 ≤ r.2 ∧ r.2 < 1 := by
  induction ds generalizing st with
  | nil => simp [h0, h1]
  | cons d ds ih =>
    simp only [List.foldl, List.sum_cons]
    obtain ⟨e, f0, f1⟩ := ih (carryStep st d) (Int.fract_nonneg _) (Int.fract_lt_one _)
    refine ⟨?_, f0, f1⟩
    rw [e]; simp only [carryStep, Int.cast_add]
    have := Int.floor_add_fract (st.2 + d); linarith

/-- **Fix 2 (persisted remainder) loses less than one cent per day**, independent of the number of
updates: the stored integer is `⌊Branch A⌋`. -/
theorem carry_eq_floor (ds : List ℚ) : branchBCarry ds = ⌊branchA ds⌋ := by
  obtain ⟨e, f0, f1⟩ := carry_invariant ds (0, 0) le_rfl zero_lt_one
  rw [branchA_eq_sum]; simp only [Int.cast_zero, zero_add] at e
  symm; rw [Int.floor_eq_iff]; unfold branchBCarry; constructor <;> linarith

/-! ## Effect on the reference budget

`FullOnlineReferenceBudgetAlgorithm` (not trending up) uses the mean of the stored completed days.
Truncation can only lower each day, so it can only lower the budget, by less than the average
number of updates per day. -/

/-- Mean of a list of days (the `series.spend.mean()` branch of `calculateOnlineReferenceBudget`). -/
def mean (xs : List ℚ) : ℚ := xs.sum / xs.length

theorem refBudget_le (days : List (List ℚ)) (h : ∀ ds ∈ days, ∀ d ∈ ds, 0 ≤ d) :
    mean (days.map (fun ds => (branchB ds : ℚ))) ≤ mean (days.map branchA) := by
  unfold mean
  simp only [List.length_map]
  apply div_le_div_of_nonneg_right _ (Nat.cast_nonneg _)
  apply List.sum_le_sum
  intro ds hds
  exact branchB_le_branchA ds (h ds hds)

/-! ## The `root_causes.md` example, executed -/

-- Five cycles of 0.4 cents (dry run A): Branch A = 2, Branch B = 0.
#eval (branchA (List.replicate 5 (2/5)), branchB (List.replicate 5 (2/5)), branchBCarry (List.replicate 5 (2/5)))
-- Three cycles of 1.6 cents: Branch A = 4.8, Branch B = 3, carry fix = 4.
#eval (branchA (List.replicate 3 (8/5)), branchB (List.replicate 3 (8/5)), branchBCarry (List.replicate 3 (8/5)))
-- Case E ad set: seven days, one stored cent among them, gives the logged 0.1429.
#eval mean [1, 0, 0, 0, 0, 0, 0]

end CaseI
