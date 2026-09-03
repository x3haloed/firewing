use crate::expert::{from_bf16, to_bf16};
use crate::hyper_connection::pytorch_inner_square_sum;

pub fn rms_norm_heads(
    input: &[u16],
    weight: &[u16],
    head_dim: usize,
    epsilon: f32,
) -> Result<Vec<u16>, String> {
    if head_dim == 0 || !input.len().is_multiple_of(head_dim) || weight.len() != head_dim {
        return Err("full-attention RMSNorm shape mismatch".to_owned());
    }
    let mut output = Vec::with_capacity(input.len());
    for head in input.chunks_exact(head_dim) {
        let float = head
            .iter()
            .map(|value| from_bf16(*value))
            .collect::<Vec<_>>();
        let inverse = (pytorch_inner_square_sum(&float) / head_dim as f32 + epsilon)
            .sqrt()
            .recip();
        output.extend(head.iter().zip(weight).map(|(value, weight)| {
            to_bf16(from_bf16(*value) * inverse * (1.0 + from_bf16(*weight)))
        }));
    }
    Ok(output)
}

pub fn apply_partial_rope(
    values: &mut [u16],
    heads: usize,
    head_dim: usize,
    cos: &[u16],
    sin: &[u16],
) -> Result<(), String> {
    let rotary_dim = cos.len();
    if heads == 0
        || head_dim == 0
        || values.len() != heads * head_dim
        || rotary_dim == 0
        || !rotary_dim.is_multiple_of(2)
        || rotary_dim > head_dim
        || sin.len() != rotary_dim
    {
        return Err("full-attention RoPE shape mismatch".to_owned());
    }
    let half = rotary_dim / 2;
    for head in values.chunks_exact_mut(head_dim) {
        for pair in 0..half {
            let first = from_bf16(head[pair]);
            let second = from_bf16(head[pair + half]);
            let cosine = from_bf16(cos[pair]);
            let sine = from_bf16(sin[pair]);
            let first_cosine = to_bf16(first * cosine);
            let second_sine = to_bf16(second * sine);
            let second_cosine = to_bf16(second * cosine);
            let first_sine = to_bf16(first * sine);
            head[pair] = to_bf16(from_bf16(first_cosine) - from_bf16(second_sine));
            head[pair + half] = to_bf16(from_bf16(second_cosine) + from_bf16(first_sine));
        }
    }
    Ok(())
}

pub fn select_qsa_blocks(scores: &[f32], block_topk: usize) -> Result<Vec<usize>, String> {
    if block_topk > scores.len() || scores.iter().any(|score| !score.is_finite()) {
        return Err("invalid QSA score vector".to_owned());
    }
    if block_topk < scores.len() {
        let mut ordered = scores.to_vec();
        ordered.sort_by(|left, right| right.total_cmp(left));
        if ordered[block_topk - 1] == ordered[block_topk] {
            return Err("QSA top-k boundary is tied".to_owned());
        }
    }
    let mut indices = (0..scores.len()).collect::<Vec<_>>();
    indices.sort_by(|left, right| scores[*right].total_cmp(&scores[*left]));
    indices.truncate(block_topk);
    Ok(indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_rms_is_independent_and_uses_one_plus_weight() {
        let input = [to_bf16(2.0), to_bf16(2.0), to_bf16(4.0), to_bf16(4.0)];
        let weight = [to_bf16(0.0), to_bf16(1.0)];
        let output = rms_norm_heads(&input, &weight, 2, 0.0).unwrap();
        assert_eq!(
            output,
            [to_bf16(1.0), to_bf16(2.0), to_bf16(1.0), to_bf16(2.0)]
        );
    }

    #[test]
    fn partial_rope_preserves_tail_and_stages_bf16_operations() {
        let mut values = [
            to_bf16(1.0),
            to_bf16(2.0),
            to_bf16(3.0),
            to_bf16(4.0),
            to_bf16(5.0),
            to_bf16(6.0),
        ];
        let tail = values[4..].to_vec();
        apply_partial_rope(&mut values, 1, 6, &[to_bf16(0.5); 4], &[to_bf16(0.25); 4]).unwrap();
        assert_eq!(&values[4..], tail);
        assert_eq!(values[0], to_bf16(-0.25));
        assert_eq!(values[2], to_bf16(1.75));
    }

    #[test]
    fn qsa_selection_is_score_ordered_and_rejects_boundary_ties() {
        assert_eq!(select_qsa_blocks(&[0.25, 2.0, 1.0], 2).unwrap(), [1, 2]);
        assert!(select_qsa_blocks(&[2.0, 1.0, 1.0], 2).is_err());
    }
}
