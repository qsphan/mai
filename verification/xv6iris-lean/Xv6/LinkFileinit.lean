/-
`fileinit`'s interface, instantiated from its proof.
-/
import Xv6.ProofFileinit

namespace Xv6

theorem Fileinit (IL : INITLOCK) : FILEINIT := fileinit_proof IL

end Xv6
