/-
`mappages`' interface, from its proof and the interface of `walk`.
-/
import Xv6.ProofMappages

namespace Xv6

/-- `mappages` meets its general (uncounted) specification, given `walk`'s. -/
theorem MappagesAny (W : WALK) : MAPPAGES_ANY := mappages_any_proof W

/-- `mappages` meets its counted specification, given `walk`'s. -/
theorem Mappages (W : WALK) : MAPPAGES := mappages_proof W

end Xv6
