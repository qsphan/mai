/-
`iinit`'s interface, instantiated from its proof.
-/
import Xv6.ProofIinit

namespace Xv6

theorem Iinit (IL : INITLOCK) (IS : INITSLEEPLOCK) : IINIT := iinit_proof IL IS

end Xv6
