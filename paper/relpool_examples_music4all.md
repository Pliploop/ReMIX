# ReMIX-B relevance-pool examples — Music4All

## chain chain_00000067 turn 1
- **source**: `vG6kmdt67f8EygFy::0-30::1`  →  **target**: `VGT0QerDi3w9OMQa::0-30::1`
- **instruction**: add screamed vocals and melodic power chords

| grade | pool type | clip | judge reason |
|---|---|---|---|
| 5 | Type_TARGET | `nuRErdzvMXqnltos::0-30::1` | The candidate caption is semantically identical to the target caption (similarity 0.9997), satisfyin |
| 5 | Type_TARGET | `ox1BMIIV9kMUXO6z::0-30::1` | The candidate caption is semantically identical to the target caption, satisfying all new constraint |
| 4 | Type_TARGET | `VGT0QerDi3w9OMQa::0-30::1` |  |
| 0 | Type_HARD_NEG | `1XqQhTYiCr9nL2zK::0-30::1` |  |
| 0 | Type_HARD_NEG | `Unhq5a0ezoDOMNoA::0-30::1` |  |

## chain chain_00000067 turn 2
- **source**: `VGT0QerDi3w9OMQa::0-30::1`  →  **target**: `nuRErdzvMXqnltos::0-30::1`
- **instruction**: crank the tempo up to a blistering fast pace

| grade | pool type | clip | judge reason |
|---|---|---|---|
| 5 | Type_TARGET | `TBsO0RBlsdDYTx0Q::0-30::1` | The candidate caption is identical to the target caption. The candidate metadata (speed: fast, vocal |
| 4 | Type_TARGET | `nuRErdzvMXqnltos::0-30::1` |  |
| 2 | Type_STRONG | `Oh2kqJ0fh76lfo7L::0-30::1` | The candidate satisfies the primary edit (fast speed) and preserves the core semantic constraints (m |
| 0 | Type_HARD_NEG | `1XqQhTYiCr9nL2zK::0-30::1` | The candidate fails the primary edit constraint by maintaining 'medium speed' instead of the request |
| 0 | Type_HARD_NEG | `VGT0QerDi3w9OMQa::0-30::1` | The candidate fails the primary edit constraint by maintaining 'medium speed' instead of the request |

## chain chain_00000067 turn 3
- **source**: `nuRErdzvMXqnltos::0-30::1`  →  **target**: `uD8ofaCfxM74nFrw::0-30::1`
- **instruction**: slow it down, more emo

| grade | pool type | clip | judge reason |
|---|---|---|---|
| 4 | Type_TARGET | `uD8ofaCfxM74nFrw::0-30::1` |  |
| 2 | Type_STRONG | `y0ZafC3G45Dp0OTh::0-30::1` | The candidate satisfies all explicit semantic constraints derived from the edit: it preserves the me |
| 2 | Type_STRONG | `1XqQhTYiCr9nL2zK::0-30::1` | The candidate satisfies the primary edit constraints: it has 'medium speed' (explicitly requested ne |
| 0 | Type_HARD_NEG | `TBsO0RBlsdDYTx0Q::0-30::1` | The candidate is the exact source clip, failing to apply any of the requested edits. It retains 'fas |
| 0 | Type_HARD_NEG | `cxvnSvQxM16K7S6Q::0-30::1` | The candidate fails to satisfy the primary edit instructions. The request explicitly asked to 'Slow  |

## chain chain_00000067 turn 4
- **source**: `uD8ofaCfxM74nFrw::0-30::1`  →  **target**: `qBZQ7ByN9yfZZr8T::0-30::1`
- **instruction**: swap the chugging riffs for harmonized leads and tremolo picking

| grade | pool type | clip | judge reason |
|---|---|---|---|
| 5 | Type_TARGET | `7tUMQmVCMvxHteKO::0-30::1` | The candidate caption is semantically identical to the target caption, satisfying all new constraint |
| 5 | Type_TARGET | `qX7dfHnoNNTtB8fJ::0-30::1` | The candidate caption is semantically identical to the target caption, satisfying all new constraint |
| 5 | Type_TARGET | `DYjyTqAuUE8XIEtp::0-30::1` | The candidate caption is semantically identical to the target caption, satisfying all new constraint |
| 0 | Type_HARD_NEG | `jm5jIKKYpstSQUCx::0-30::1` |  |
| 0 | Type_HARD_NEG | `k0rpXNDQ9yOTfU0I::0-30::1` | The candidate caption explicitly states 'no vocals are present', which directly contradicts the expl |

## chain chain_00000067 turn 5
- **source**: `qBZQ7ByN9yfZZr8T::0-30::1`  →  **target**: `Ynd6P8T8TUyBNmhB::0-30::1`
- **instruction**: switch to folk viking metal, driving rhythm

| grade | pool type | clip | judge reason |
|---|---|---|---|
| 5 | Type_TARGET | `k4vuVtUYe0ZkxFUJ::0-30::1` | The candidate caption is semantically identical to the target caption, describing the same genre shi |
| 5 | Type_TARGET | `UR5XuKkOTqbuicWN::0-30::1` | The candidate caption is semantically identical to the target caption (similarity 0.9978), describin |
| 4 | Type_TARGET | `Ynd6P8T8TUyBNmhB::0-30::1` |  |
| 0 | Type_HARD_NEG | `uOA8zaba7H9YpHaq::0-30::1` | The candidate fails to execute the primary edit. The instruction explicitly requires switching FROM  |
| 0 | Type_HARD_NEG | `3bkZemgeYqXbIQoT::0-30::1` | The candidate is the source clip itself (or a near-identical version), failing to apply the requeste |

## chain chain_00000149 turn 1
- **source**: `BjRtmDm6oZVC6nJZ::0-30::1`  →  **target**: `PM1qmXIpCY6yFdTV::0-30::1`
- **instruction**: flip to 70s soul-pop with a male singer

| grade | pool type | clip | judge reason |
|---|---|---|---|
| 4 | Type_TARGET | `PM1qmXIpCY6yFdTV::0-30::1` |  |
| 0 | Type_HARD_NEG | `KT46ZolxgWRkUtPd::0-30::1` | The candidate completely fails the primary edit instruction. The request was to shift to a '70s soul |
| 0 | Type_HARD_NEG | `uwrVajJay2Qce20i::0-30::1` | The candidate is a high-energy pop song with female vocals, which directly contradicts the requested |
| 0 | Type_HARD_NEG | `KT46ZolxgWRkUtPd::0-30::1` | The candidate completely fails the primary edit instruction. The request was to shift to a '70s soul |
| 0 | Type_HARD_NEG | `uwrVajJay2Qce20i::0-30::1` | The candidate is a high-energy pop song with female vocals, which directly contradicts the requested |

