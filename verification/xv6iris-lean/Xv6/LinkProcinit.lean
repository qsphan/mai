/-
`procinit`'s interface, instantiated from its proof.
-/
import Xv6.ProofProcinit

namespace Xv6

theorem Procinit (IL : INITLOCK) : PROCINIT := procinit_proof IL

end Xv6
