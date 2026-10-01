/-
`printkinit`'s interface, instantiated from its proof.
-/
import Xv6.ProofPrintkinit

namespace Xv6

theorem Printkinit (IL : INITLOCK) : PRINTKINIT := printkinit_proof IL

end Xv6
