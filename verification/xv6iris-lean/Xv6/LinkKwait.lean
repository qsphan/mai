/-
Link `kwait`: the proof instance clients import.  `kwait` calls `myproc`,
`acquire` / `release` (on `wait_lock` and each child `pp->lock`),
`copyout`, `freeproc`, `killed`, `sleep_prepare` and `sleep`; the
interfaces stay parameters here, so a client may close them with the
linked ones or with its own.

`kwait_proof` is complete and `sorry`-free (see `Xv6/ProofKwait.lean`).
-/
import Xv6.ProofKwait

namespace Xv6

/-- The `kwait` interface, given its callees. -/
theorem Kwait (MP : MYPROC) (AC : ACQUIRE) (RE : RELEASE) (CO : COPYOUT)
    (FP : FREEPROC) (KL : KILLED) (SP : SLEEP_PREPARE) (SL : SLEEP) : KWAIT :=
  kwait_proof MP AC RE CO FP KL SP SL

end Xv6
