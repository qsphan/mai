/-
`uvmunmap`'s interfaces (all three contracts), from its proof and the
interfaces of `walk` (non-allocating) and `kfree`.
-/
import Xv6.ProofUvmunmap

namespace Xv6

/-- `uvmunmap` meets both of its specifications, given `walk`'s and `kfree`'s. -/
theorem Uvmunmap (W : WALK_NOALLOC) (KF : KFREE) : UVMUNMAP := uvmunmap_proof W KF

/-- `uvmunmap` also meets the bare-table freeing contract (`uvmfree`'s
caller altitude), given the same two interfaces. -/
theorem UvmunmapBare (W : WALK_NOALLOC) (KF : KFREE) : UVMUNMAP_BARE := uvmunmap_bare_proof W KF

end Xv6
