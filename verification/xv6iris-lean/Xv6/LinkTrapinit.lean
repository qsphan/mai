/-
`trapinit`'s interface, instantiated from its proof.
-/
import Xv6.ProofTrapinit

namespace Xv6

theorem Trapinit (IL : INITLOCK) : TRAPINIT := trapinit_proof IL

end Xv6
