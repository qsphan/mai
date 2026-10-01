/-
Link `sleep`: the proof instance clients import.  `sleep` calls `myproc`,
`acquire`, `release` and `sched`; the interfaces stay parameters here, so a
client may close them with the linked ones (`LinkMyproc`, `LinkAcquire`,
`LinkRelease`, `LinkSched`) or with its own.
-/
import Xv6.ProofSleep

namespace Xv6

/-- The proved `sleep` interface, given `myproc`, `acquire`, `release` and
`sched`. -/
theorem Sleep (MP : MYPROC) (AC : ACQUIRE) (RE : RELEASE) (SC : SCHED) : SLEEP :=
  sleep_proof MP AC RE SC

end Xv6
