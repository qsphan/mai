/-
`kinit`'s interface, instantiated from its proof.
-/
import Xv6.ProofKinit

namespace Xv6

theorem Kinit (IL : INITLOCK) (FR : FREERANGE) : KINIT := kinit_proof IL FR

end Xv6
