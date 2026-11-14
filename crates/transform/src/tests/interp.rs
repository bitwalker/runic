//! A tiny interpreter over the printed text of a function, for before/after comparisons.
//!
//! It understands the operations the generators of these tests emit: `arith.constant`, `arith.add`,
//! `arith.cmpi`, `mem.load`/`store`/`alloca`/`memcpy`/`memmove`/`memset`, `test.load`/`store`,
//! `test.effects`, `test.touch_local`, `builtin.load_local`/`store_local`, `builtin.call` (the
//! callee's name decides what it does), `cf.assert`, `cf.br`, `cf.cond_br`, `builtin.ret`,
//! `scf.if` and `scf.yield`. Anything else is `Err(unsupported)`.
//!
//! The semantics are deliberately adversarial but consistent:
//! * `mem` memory and `test` memory are two maps from address to value (two pointer arguments
//!   may hold the same address);
//! * a call to `@unknown*` or `@writes*`, and a `test.effects` with unknown effects, scramble both
//!   memories; they never touch the caller's locals;
//! * `@reads*` returns a digest of its arguments and both memories; `@pure*` of its arguments;
//! * `@frees*` scrambles the cell its argument addresses, in both memories;
//! * a `cf.assert` of 0 traps;
//! * `mem.ptrtoint`, `mem.inttoptr` and `mem.ptrcast` give their operand: a pointer is as wide as
//!   an `isize`. The functions these tests run state no data layout, so no pass may fold a cast
//!   round trip on that assumption, and none does.
//!
//! What is observed: how the function ends (returned values, trap, out of fuel), the trace of
//! effectful calls with their arguments, and both final memories.

use alloc::{
    collections::BTreeMap,
    format,
    string::{String, ToString},
    vec::Vec,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node {
    Op {
        results: Vec<String>,
        name: String,
        text: String,
    },
    If {
        results: Vec<String>,
        cond: String,
        then_ops: Vec<Node>,
        else_ops: Vec<Node>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub name: String,
    pub params: Vec<String>,
    pub ops: Vec<Node>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Func {
    pub params: Vec<String>,
    pub blocks: Vec<Block>,
}

/// Every `%name` in `s`, in order.
pub fn values(s: &str) -> Vec<String> {
    let bytes = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let start = i;
            i += 1;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            out.push(s[start..i].to_string());
        } else {
            i += 1;
        }
    }
    out
}

fn split_op(line: &str) -> (Vec<String>, String, String) {
    let line = line.trim().trim_end_matches(';');
    let (results, rest) = match line.find(" = ") {
        Some(at) if line.starts_with('%') => (values(&line[..at]), &line[at + 3..]),
        _ => (Vec::new(), line),
    };
    let name = rest.split([' ', ';']).next().unwrap().to_string();
    let text = rest[name.len()..].trim().to_string();
    (results, name, text)
}

fn parse_nodes(lines: &[&str], i: &mut usize) -> Result<Vec<Node>, String> {
    let mut nodes = Vec::new();
    while *i < lines.len() {
        let line = lines[*i].trim();
        if line.is_empty() {
            *i += 1;
            continue;
        }
        if line.starts_with('}') || line.starts_with('^') {
            break;
        }
        let (results, name, text) = split_op(line);
        if line.ends_with('{') {
            if name != "scf.if" {
                return Err(format!("unsupported region operation {name}"));
            }
            let cond = values(&text)[0].clone();
            *i += 1;
            let then_ops = parse_nodes(lines, i)?;
            if !lines[*i].trim().starts_with("} else {") {
                return Err(format!("expected else, got {}", lines[*i]));
            }
            *i += 1;
            let else_ops = parse_nodes(lines, i)?;
            if !lines[*i].trim().starts_with('}') {
                return Err(format!("expected end of if, got {}", lines[*i]));
            }
            *i += 1;
            nodes.push(Node::If {
                results,
                cond,
                then_ops,
                else_ops,
            });
            continue;
        }
        *i += 1;
        nodes.push(Node::Op {
            results,
            name,
            text,
        });
    }
    Ok(nodes)
}

/// Parses the printed text of one `builtin.function`.
pub fn parse_function(text: &str) -> Result<Func, String> {
    let lines = text.lines().collect::<Vec<_>>();
    let header = lines[0];
    // the header is `builtin.function public extern("C") @f(..)`: skip `extern("C")`
    let at = header.find(" @").unwrap();
    let params_start = at + header[at..].find('(').unwrap();
    let close = matching_paren(header, params_start);
    let params = values(&header[params_start..close]);
    let mut blocks = Vec::new();
    let mut i = 1;
    let mut current = Block {
        name: "entry".to_string(),
        params: Vec::new(),
        ops: Vec::new(),
    };
    loop {
        current.ops = parse_nodes(&lines, &mut i)?;
        blocks.push(current);
        if i >= lines.len() {
            break;
        }
        let line = lines[i].trim();
        if line.starts_with('^') {
            let name = line[..line.find(['(', ':']).unwrap()].to_string();
            current = Block {
                name,
                params: values(line),
                ops: Vec::new(),
            };
            i += 1;
        } else {
            break;
        }
    }
    Ok(Func { params, blocks })
}

fn matching_paren(s: &str, open: usize) -> usize {
    let mut depth = 0;
    for (i, c) in s[open..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return open + i;
                }
            }
            _ => (),
        }
    }
    panic!("unbalanced: {s}")
}

pub fn mix(a: u64, b: u64) -> u64 {
    let mut x = a ^ b.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn hash_str(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| mix(h, u64::from(b)))
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Memory {
    epoch: u64,
    cells: BTreeMap<u64, u64>,
}

impl Memory {
    fn read(&self, addr: u64) -> u64 {
        self.cells
            .get(&addr)
            .copied()
            .unwrap_or_else(|| mix(mix(self.epoch, addr), 0x11) & 0xffff_ffff)
    }

    fn write(&mut self, addr: u64, value: u64) {
        self.cells.insert(addr, value);
    }

    fn scramble(&mut self, with: u64) {
        self.cells.clear();
        self.epoch = mix(self.epoch, with);
    }

    fn digest(&self) -> u64 {
        self.cells
            .iter()
            .fold(self.epoch, |h, (addr, value)| mix(mix(h, *addr), *value))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum End {
    Ret(Vec<u64>),
    Trap(String),
    Fuel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome {
    pub end: End,
    pub trace: Vec<String>,
    pub mem: Memory,
    pub test_mem: Memory,
}

enum Flow {
    Next,
    Yield(Vec<u64>),
    Ret(Vec<u64>),
    Trap(String),
    Fuel,
    Jump(String, Vec<u64>),
}

struct Machine {
    env: BTreeMap<String, u64>,
    mem: Memory,
    test_mem: Memory,
    locals: BTreeMap<u64, u64>,
    trace: Vec<String>,
    fuel: u32,
    touches: u64,
}

impl Machine {
    fn get(&self, name: &str) -> Result<u64, String> {
        self.env
            .get(name)
            .copied()
            .ok_or_else(|| format!("use of undefined value {name}"))
    }

    fn args(&self, text: &str) -> Result<Vec<u64>, String> {
        values(text).iter().map(|v| self.get(v)).collect()
    }

    fn local_index(text: &str) -> u64 {
        let at = text.find("local_variable<").expect("a local") + "local_variable<".len();
        let end = at + text[at..].find(',').unwrap();
        text[at..end].trim().parse().unwrap()
    }

    fn exec_nodes(&mut self, nodes: &[Node]) -> Result<Flow, String> {
        for node in nodes {
            match node {
                Node::If {
                    results,
                    cond,
                    then_ops,
                    else_ops,
                } => {
                    if self.fuel == 0 {
                        return Ok(Flow::Fuel);
                    }
                    self.fuel -= 1;
                    let taken = if self.get(cond)? & 1 == 1 {
                        then_ops
                    } else {
                        else_ops
                    };
                    match self.exec_nodes(taken)? {
                        Flow::Yield(vals) => {
                            if vals.len() != results.len() {
                                return Err("yield arity".to_string());
                            }
                            for (r, v) in results.iter().zip(vals) {
                                self.env.insert(r.clone(), v);
                            }
                        }
                        Flow::Next => return Err("region fell off its end".to_string()),
                        other => return Ok(other),
                    }
                }
                Node::Op {
                    results,
                    name,
                    text,
                } => match self.exec_op(results, name, text)? {
                    Flow::Next => (),
                    other => return Ok(other),
                },
            }
        }
        Ok(Flow::Next)
    }

    fn exec_op(&mut self, results: &[String], name: &str, text: &str) -> Result<Flow, String> {
        let operand_text = text.split(" : ").next().unwrap();
        let set = |this: &mut Self, v: u64| {
            if let Some(r) = results.first() {
                this.env.insert(r.clone(), v);
            }
        };
        match name {
            "arith.constant" => {
                let lit = operand_text.trim();
                let v: i128 = lit.parse().map_err(|_| format!("constant {lit}"))?;
                set(self, (v as u64) & 0xffff_ffff);
            }
            "arith.add" => {
                let a = self.args(operand_text)?;
                set(self, a[0].wrapping_add(a[1]) & 0xffff_ffff);
            }
            "arith.cmpi" => {
                let pred = operand_text.split_whitespace().next().unwrap();
                let a = self.args(operand_text)?;
                let (x, y) = (a[0] as u32, a[1] as u32);
                let r = match pred {
                    "eq" => x == y,
                    "ne" => x != y,
                    "ult" => x < y,
                    "slt" => (x as i32) < (y as i32),
                    other => return Err(format!("unsupported predicate {other}")),
                };
                set(self, u64::from(r));
            }
            "mem.load" => {
                let a = self.args(operand_text)?;
                set(self, self.mem.read(a[0]));
            }
            "mem.store" => {
                let a = self.args(operand_text)?;
                self.mem.write(a[1], a[0]);
            }
            "mem.alloca" => {
                let addr = 0xa000_0000_0000 | (hash_str(&results[0]) & 0xffff_fff0);
                set(self, addr);
            }
            "mem.ptrcast" | "mem.ptrtoint" | "mem.inttoptr" => {
                let a = self.args(operand_text)?;
                set(self, a[0]);
            }
            "mem.memcpy" | "mem.memmove" => {
                let a = self.args(operand_text)?;
                if a[2] != 0 {
                    let v = self.mem.read(a[1]);
                    self.mem.write(a[0], v);
                }
            }
            "mem.memset" => {
                let a = self.args(operand_text)?;
                if a[2] != 0 {
                    self.mem.write(a[0], (a[1] & 0xff) * 0x0101_0101);
                }
            }
            "test.load" => {
                let a = self.args(operand_text)?;
                set(self, self.test_mem.read(a[0]));
            }
            "test.store" => {
                let a = self.args(operand_text)?;
                self.test_mem.write(a[0], a[1]);
            }
            "test.effects" => {
                if text.contains("bool<false>") {
                    self.trace.push("test.effects".to_string());
                    let n = self.trace.len() as u64;
                    self.mem.scramble(n);
                    self.test_mem.scramble(n);
                    set(self, mix(n, 0x77) & 0xffff_ffff);
                } else {
                    set(self, mix(self.mem.digest(), self.test_mem.digest()) & 0xffff_ffff);
                }
            }
            "test.touch_local" => {
                let k = Self::local_index(text);
                self.touches += 1;
                self.locals.insert(k, mix(mix(0x7071, k), self.touches) & 0xffff_ffff);
            }
            "builtin.load_local" => {
                let k = Self::local_index(text);
                let v = self.locals.get(&k).copied().unwrap_or(mix(0x10ca1, k) & 0xffff_ffff);
                set(self, v);
            }
            "builtin.store_local" => {
                let k = Self::local_index(text);
                let a = self.args(text.split("<{").next().unwrap())?;
                self.locals.insert(k, a[0]);
            }
            "builtin.call" => {
                let callee_start = text.find('@').unwrap();
                let open = callee_start + text[callee_start..].find('(').unwrap();
                let callee = &text[callee_start..open];
                let close = matching_paren(text, open);
                let a = self.args(&text[open..close])?;
                let args_hash = a.iter().fold(hash_str(callee), |h, v| mix(h, *v));
                if callee.starts_with("@pure") {
                    set(self, args_hash & 0xffff_ffff);
                } else if callee.starts_with("@reads") {
                    let h = mix(mix(args_hash, self.mem.digest()), self.test_mem.digest());
                    set(self, h & 0xffff_ffff);
                } else if callee.starts_with("@frees") {
                    self.trace.push(format!("{callee}{a:?}"));
                    let n = self.trace.len() as u64;
                    self.mem.write(a[0], mix(n, 0xf4ee) & 0xffff_ffff);
                    self.test_mem.write(a[0], mix(n, 0xf4ef) & 0xffff_ffff);
                } else if callee.starts_with("@writes") || callee.starts_with("@unknown") {
                    self.trace.push(format!("{callee}{a:?}"));
                    let n = mix(self.trace.len() as u64, args_hash);
                    self.mem.scramble(n);
                    self.test_mem.scramble(n);
                    set(self, mix(n, 0x33) & 0xffff_ffff);
                } else {
                    return Err(format!("unsupported callee {callee}"));
                }
            }
            "cf.assert" => {
                let a = self.args(operand_text)?;
                if a[0] & 1 == 0 {
                    let message = text.find('"').map(|at| &text[at..]).unwrap_or(text);
                    return Ok(Flow::Trap(message.to_string()));
                }
            }
            "cf.br" => {
                let (target, args) = self.successor(text.trim())?;
                return Ok(Flow::Jump(target, args));
            }
            "cf.cond_br" => {
                // `%c ^a(..), ^b(..) : (i1)`
                let cond = values(text)[0].clone();
                let first = text.find('^').unwrap();
                let rest = &text[first..];
                let split = top_level_comma(rest);
                let (then_text, else_text) = (&rest[..split], rest[split + 1..].trim());
                let else_text = strip_trailing_types(else_text);
                let chosen = if self.get(&cond)? & 1 == 1 {
                    then_text
                } else {
                    else_text
                };
                let (target, args) = self.successor(chosen.trim())?;
                return Ok(Flow::Jump(target, args));
            }
            "builtin.ret" => return Ok(Flow::Ret(self.args(operand_text)?)),
            "scf.yield" => return Ok(Flow::Yield(self.args(operand_text)?)),
            other => return Err(format!("unsupported operation {other}")),
        }
        Ok(Flow::Next)
    }

    /// `^name` or `^name(%a : t, ..)`.
    fn successor(&self, text: &str) -> Result<(String, Vec<u64>), String> {
        match text.find('(') {
            Some(open) => {
                let close = matching_paren(text, open);
                let inner = &text[open + 1..close];
                let mut args = Vec::new();
                for part in inner.split(',') {
                    let v = values(part);
                    if let Some(v) = v.first() {
                        args.push(self.get(v)?);
                    }
                }
                Ok((text[..open].trim().to_string(), args))
            }
            None => Ok((text.trim().to_string(), Vec::new())),
        }
    }
}

fn top_level_comma(s: &str) -> usize {
    let mut depth = 0;
    for (i, c) in s.char_indices() {
        match c {
            '(' | '<' => depth += 1,
            ')' | '>' => depth -= 1,
            ',' if depth == 0 => return i,
            _ => (),
        }
    }
    panic!("no comma in {s}")
}

/// Drops a trailing ` : (..)` that is not inside parentheses.
fn strip_trailing_types(s: &str) -> &str {
    let mut depth = 0;
    let bytes = s.as_bytes();
    for i in 0..bytes.len() {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => depth -= 1,
            b' ' if depth == 0 && s[i..].starts_with(" : ") => return &s[..i],
            _ => (),
        }
    }
    s
}

/// Runs `func` on `args` (one value per parameter).
pub fn run(func: &Func, args: &[u64], fuel: u32) -> Result<Outcome, String> {
    assert_eq!(func.params.len(), args.len());
    let mut machine = Machine {
        env: BTreeMap::new(),
        mem: Memory::default(),
        test_mem: Memory::default(),
        locals: BTreeMap::new(),
        trace: Vec::new(),
        fuel,
        touches: 0,
    };
    for (p, a) in func.params.iter().zip(args) {
        machine.env.insert(p.clone(), *a);
    }
    let mut block = 0;
    let end = loop {
        match machine.exec_nodes(&func.blocks[block].ops)? {
            Flow::Next => return Err("block fell off its end".to_string()),
            Flow::Yield(_) => return Err("yield outside a region".to_string()),
            Flow::Ret(vals) => break End::Ret(vals),
            Flow::Trap(msg) => break End::Trap(msg),
            Flow::Fuel => break End::Fuel,
            Flow::Jump(target, vals) => {
                if machine.fuel == 0 {
                    break End::Fuel;
                }
                machine.fuel -= 1;
                block = func
                    .blocks
                    .iter()
                    .position(|b| b.name == target)
                    .ok_or_else(|| format!("no block {target}"))?;
                let params = &func.blocks[block].params;
                if params.len() != vals.len() {
                    return Err(format!("block {target} arity"));
                }
                for (p, v) in params.iter().zip(vals) {
                    machine.env.insert(p.clone(), v);
                }
            }
        }
    };
    Ok(Outcome {
        end,
        trace: machine.trace,
        mem: machine.mem,
        test_mem: machine.test_mem,
    })
}
