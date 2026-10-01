/-
`initsleeplock`'s interface, instantiated from its proof.
-/
import Xv6.ProofInitsleeplock

namespace Xv6

theorem Initsleeplock (IL : INITLOCK) : INITSLEEPLOCK := initsleeplock_proof IL

end Xv6
