/-
`freewalk`'s interface, from its proof and the interface of `kfree`.
-/
import Xv6.ProofFreewalk

namespace Xv6

/-- `freewalk` meets its specification, given `kfree`'s. -/
theorem Freewalk (KF : KFREE) : FREEWALK := freewalk_proof KF

end Xv6
