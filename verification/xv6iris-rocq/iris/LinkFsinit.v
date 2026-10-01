(* LinkFsinit.v -- the only file where fsinit's proof meets its callees'.
   THE LAST LINK OF fs.c: with this module, all twenty-four of fs.c's
   functions are PROVEN AND LINKED.

   All five callees are proven, so nothing here is assumed:

   - bread / brelse arrive through their own Link files, the same instances
     balloc, ialloc, iupdate and ireclaim use;
   - memmove arrives through LinkMemmove.v -- the same leaf instance ilock,
     iupdate, install_trans, end_op, copyin and copyout drive;
   - initlog arrives through LinkInitlog.v (initlock / bread / brelse /
     install_trans / write_head, sealed inside it);
   - ireclaim arrives through LinkIreclaim.v (bread / brelse / iget /
     begin_op / ilock / iunlock / iput / end_op, sealed inside it).

   fsinit's ONE dead arm is refuted inside the proof, so no panic contract is
   instantiated here:

   - the [bne a4,a5] at +0x40 -- the C source's
     [if(sb.magic != FSMAGIC) panic("invalid file system")], which is a REAL
     panic and not one of the [printk] arms balloc and ialloc have -- is
     refuted from the contract's [bv_unsigned v_magic = FSMAGIC], an IMAGE
     premise about the 32 bytes mkfs wrote into block 1.  That is the whole
     reason SpecFsinit.v states its geometry as claims about [sb_image].

   The [kernel_data] / [panic_env] the contract takes are threaded to all
   five callees, whose own panic arms are discharged against [Panic].


   THE BOOT CLIENT'S SIDE OF THE BARGAIN, in one place: fsinit needs
   [bslots 35] ((LOGBLOCKS + 2) + 2 + 1), the raw 32 bytes of .bss at
   [&sb], block 1's client half at the mkfs image, initlog's whole raw
   [struct log] bundle and FsBlocks material, the icache's four persistent
   things out of [IcacheBoot.icache_boot], and the image premises SpecFsinit.v
   names.  It gets back the eight typed superblock cells, the log context and
   [bslots 3] -- block 1's half does NOT come back (durable-disk lane C-3a):
   it is spent into initlog's [SbPark] park and rides out inside the
   [LogInv.log_ctx]. *)
Require Import LinkBread LinkMemmove LinkBrelse LinkInitlog LinkIreclaim
                ProofFsinit.

Module Fsinit := FsinitProof Bread Memmove Brelse Initlog Ireclaim.
