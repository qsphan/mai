/-
`kvmmap`'s interface, instantiated from its proof.
-/
import Xv6.ProofKvmmap

namespace Xv6

theorem Kvmmap (MP : MAPPAGES) : KVMMAP := kvmmap_proof MP

end Xv6
