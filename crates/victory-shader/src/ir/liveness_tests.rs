//! Independently generated control-flow graphs; no title data.
use super::*;

fn next(seed: &mut u32) -> usize {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed as usize
}

fn check(blocks: &mut [Block], uses: &[RegSet], defs: &[RegSet]) {
    link(blocks);
    let order = reverse_postorder(blocks);
    let empty = [0; REGS.div_ceil(64)];
    let mut inn = vec![empty; blocks.len()];
    let mut out = vec![empty; blocks.len()];
    // Original full-sweep solver, retained only as a test oracle.
    loop {
        let mut changed = false;
        for &b in order.iter().rev() {
            let mut next_out = empty;
            for s in blocks[b].exit.successors() {
                for (o, i) in next_out.iter_mut().zip(&inn[s]) {
                    *o |= i;
                }
            }
            let next_in = std::array::from_fn(|w| uses[b][w] | next_out[w] & !defs[b][w]);
            changed |= next_in != inn[b] || next_out != out[b];
            inn[b] = next_in;
            out[b] = next_out;
        }
        if !changed {
            break;
        }
    }
    assert_eq!(solve_liveness(blocks, &order, uses, defs), (inn, out));
}

#[test]
fn generated_graphs_match_full_sweeps() {
    let mut seed = 0x7812_abcd;
    for _ in 0..4096 {
        let n = next(&mut seed) % 48 + 1;
        let mut blocks: Vec<_> = (0..n)
            .map(|offset| {
                let exit = match next(&mut seed) % 4 {
                    0 => Exit::End,
                    1 => Exit::Jump(next(&mut seed) % n),
                    _ => Exit::Branch {
                        taken: next(&mut seed) % n,
                        not_taken: next(&mut seed) % n,
                    },
                };
                Block { offset, preds: Vec::new(), phis: Vec::new(), insts: Vec::new(), exit }
            })
            .collect();
        let mut masks = || {
            (0..n)
                .map(|_| {
                    let mut set = [0; REGS.div_ceil(64)];
                    for _ in 0..8 {
                        insert(&mut set, reg_at(next(&mut seed) % REGS));
                    }
                    set
                })
                .collect::<Vec<_>>()
        };
        let uses = masks();
        let defs = masks();
        check(&mut blocks, &uses, &defs);
    }
}

#[test]
fn long_chain_with_backedge_matches_full_sweeps() {
    let n = 2048;
    let mut blocks: Vec<_> = (0..n)
        .map(|offset| Block {
            offset,
            preds: Vec::new(),
            phis: Vec::new(),
            insts: Vec::new(),
            exit: if offset + 1 < n { Exit::Jump(offset + 1) } else { Exit::Jump(n / 2) },
        })
        .collect();
    let mut uses = vec![[0; REGS.div_ceil(64)]; n];
    let mut defs = uses.clone();
    insert(&mut uses[n / 2], Reg::V(255));
    insert(&mut uses[n - 1], Reg::Scc);
    insert(&mut defs[n / 4], Reg::V(255));
    check(&mut blocks, &uses, &defs);
}
