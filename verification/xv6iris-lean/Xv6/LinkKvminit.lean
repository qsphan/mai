/-
`kvminit`'s interface, instantiated from its proof.
-/
import Xv6.ProofKvminit

namespace Xv6

theorem Kvminit (KV : KVMMAKE) : KVMINIT := kvminit_proof KV

end Xv6
