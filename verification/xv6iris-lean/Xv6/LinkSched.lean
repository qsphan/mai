/-
Link `sched`: the sealed proof instance clients import.
-/
import Xv6.ProofSched
import Xv6.LinkSwtch
import Xv6.LinkMyproc
import Xv6.LinkHolding

namespace Xv6

/-- The proved `sched` interface. -/
theorem Sched : SCHED := sched_proof Swtch Myproc Holding

end Xv6
