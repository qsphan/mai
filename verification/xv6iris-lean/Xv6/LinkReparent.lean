/-
`reparent` meets its interface, given `wakeup`'s.
-/
import Xv6.ProofReparent

namespace Xv6

/-- `reparent`'s contract holds, given `wakeup`'s. -/
theorem Reparent (WK : WAKEUP) : REPARENT := reparent_proof WK

end Xv6
