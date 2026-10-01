/-
`binit`'s interface, instantiated from its proof.
-/
import Xv6.ProofBinit

namespace Xv6

theorem Binit (IL : INITLOCK) (IS : INITSLEEPLOCK) : BINIT := binit_proof IL IS

end Xv6
