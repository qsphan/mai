/-
`freerange`'s interface, instantiated from its proof.
-/
import Xv6.ProofFreerange

namespace Xv6

theorem Freerange (KF : KFREE) : FREERANGE := freerange_proof KF

end Xv6
