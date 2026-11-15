//! One module per dialect of the workspace that uses each of the dialect's operations, for the
//! tests of printing and parsing (`generic_form.rs`). An operation without a text of its own is
//! written in the generic form.

/// The modules, each with the name of its dialect.
pub const DIALECT_MODULES: [(&str, &str); 8] = [
    (
        "arith",
        r#"builtin.function public extern("C") @arith(%a: i32, %b: i32, %w: i64) -> i32 {
    %c = arith.constant 7 : i32;
    %0 = arith.add %a, %b;
    %1, %2 = arith.add_overflowing %a, %b <{ signedness = #builtin.signedness<unsigned> }>;
    %3 = arith.sub %0, %c;
    %4, %5 = arith.sub_overflowing %a, %b <{ signedness = #builtin.signedness<signed> }>;
    %6 = arith.mul %3, %5;
    %7, %8 = arith.mul_overflowing %a, %b <{ signedness = #builtin.signedness<unsigned> }>;
    %9 = arith.neg %6;
    %10 = arith.udiv %9, %c;
    %11 = arith.sdiv %10, %c;
    %12 = arith.urem %11, %c;
    %13 = arith.srem %12, %c;
    %14, %15 = arith.udivrem %13, %c;
    %16, %17 = arith.sdivrem %14, %c;
    %18 = arith.min %16, %17 <{ signedness = #builtin.signedness<signed> }>;
    %19 = arith.max %18, %b <{ signedness = #builtin.signedness<unsigned> }>;
    %20 = arith.band %19, %a;
    %21 = arith.bor %20, %b;
    %22 = arith.bxor %21, %a;
    %23 = arith.bnot %22;
    %24 = arith.shl %23, %b;
    %25 = arith.shr %24, %b;
    %26 = arith.ashr %25, %b;
    %27 = arith.rotl %26, %b;
    %28 = arith.rotr %27, %b;
    %29 = arith.popcnt %28;
    %30 = arith.clz %29;
    %31 = arith.ctz %30;
    %32 = arith.clo %31;
    %33 = arith.cto %32;
    %34 = arith.zext %33 <{ ty = #builtin.type<i64> }>;
    %35 = arith.sext %33 <{ ty = #builtin.type<i64> }>;
    %36 = arith.trunc %w <{ ty = #builtin.type<i32> }>;
    %37 = arith.zext_or_trunc %33 <{ ty = #builtin.type<isize> }>;
    %38 = arith.sext_or_trunc %37 <{ ty = #builtin.type<i32> }>;
    %39 = arith.cmpi ult %34, %35;
    %40 = arith.add %36, %38;
    builtin.ret %40 : (i32);
};"#,
    ),
    (
        "builtin",
        r#"builtin.world {
    builtin.component private @c { };
    builtin.module public @m {
        builtin.global_variable public @data : array<i8; 2> {
            %0 = builtin.bytes "0201" : array<i8; 2>;
        };
        builtin.segment <{ offset = #builtin.i64<16>, data = #builtin.bytes<"cc3f"> }>;
        builtin.function public extern("C") @callee(i32) -> i32;
        builtin.function_alias private @alias -> @callee;
        builtin.function public extern("C") @narrow() -> i7 {
            builtin.ret_imm -1 : i7;
        };
        builtin.function public extern("C") @f(%a: i32) -> i32 locals(i32) {
            builtin.store_local 0, %a;
            %1 = builtin.load_local 0 : i32;
            %2 = builtin.call @alias(%1) : extern("C") (i32) -> i32;
            %3 = builtin.function_ref @callee : (i32) -> i32;
            %4 = builtin.call_indirect %3(%2) : extern("C") (i32) -> i32;
            %5 = builtin.global_symbol @data : ptr<array<i8; 2>>;
            %6 = builtin.unrealized_conversion_cast %4 : i32 to i32;
            builtin.ret %6 : (i32);
        };
    };
};"#,
    ),
    (
        "cf",
        r#"builtin.function public extern("C") @cf(%c: i1, %s: i32, %a: i32, %b: i32) -> i32 {
    cf.assert %c, "c holds";
    %0 = cf.select %c, %a, %b;
    cf.cond_br %c ^left(%0 : i32), ^right : (i1);
^left(%x: i32):
    cf.br ^join(%x : i32);
^right:
    cf.switch %s [#builtin.i32<1> -> ^join(%a : i32), #builtin.i32<-2> -> ^join(%b : i32)], ^join(%s : i32) : (i32);
^join(%r: i32):
    builtin.ret %r : (i32);
};"#,
    ),
    (
        "di",
        r#"builtin.function public extern("C") @di(%a: i32) -> i32 locals(i32) {
    di.debug_declare <{ variable = #di.variable<{ name = "x", file = "x.rs", line = 1 }>, expression = #di.expression<[DI_OP_frame_base(local_slot, 0+0)]> }>;
    di.debug_value %a <{ variable = #di.variable<{ name = "a", file = "x.rs", line = 2, column = 1 }>, expression = #di.expression<[]> }> : (i32);
    di.debug_kill <{ variable = #di.variable<{ name = "a", file = "x.rs", line = 2, column = 1 }> }>;
    builtin.ret %a : (i32);
};"#,
    ),
    (
        "mem",
        r#"builtin.function public extern("C") @mem(%p: ptr<i32>, %q: ptr<struct<i32, array<i8; 4>>>, %o: ptr, %i: isize, %n: isize, %b: i8) -> i32 {
    %0 = mem.alloca i32;
    %1 = mem.load %p;
    mem.store %1, %0;
    %2 = mem.load %o : i32;
    %3 = mem.gep %q, %i, [1, 2];
    %4 = mem.ptrcast %3 : ptr<i32>;
    %5 = mem.ptrtoint %4;
    %6 = mem.inttoptr %5 : ptr<i8>;
    mem.memcpy %6, %6, %n;
    mem.memmove %6, %6, %n;
    mem.memset %6, %b, %n;
    %7 = mem.sizeof struct<i32, i8>;
    %8 = mem.alignof i64;
    %9 = mem.offsetof struct<i32, i8>, [1];
    builtin.ret %2 : (i32);
};"#,
    ),
    (
        "scf",
        r#"builtin.function public extern("C") @scf(%n: i32, %c: i1, %s: i32) -> i32 {
    %0 = arith.constant 0 : i32;
    %1 = scf.while %0 before {
    ^before(%i: i32):
        %2 = arith.cmpi ult %i, %n;
        scf.condition %2, %i : (i1, i32);
    } after {
    ^after(%j: i32):
        %3 = scf.if %c then {
            scf.continue 1 %j : (i32);
        } else {
            scf.break 1 %j : (i32);
        } : (i1) -> (i32);
        scf.yield %3 : (i32);
    } : (i32) -> (i32);
    %4 = scf.index_switch %s
    case 1 {
        scf.yield %1 : (i32);
    }
    default {
        scf.yield %0 : (i32);
    } : (i32);
    builtin.ret %4 : (i32);
};"#,
    ),
    (
        "test",
        r#"builtin.module public @test {
    builtin.function public extern("C") @callee(i32) -> i32;
    builtin.function public extern("C") @other(i32) -> i32;
    builtin.function public extern("C") @f(%a: i32, %w: i128, %p: ptr<i32>, %s: i8) -> i32 locals(i32) {
        %0 = test.constant 7 : i32;
        %1 = test.add %a, %0;
        %2 = test.mul %1, %a;
        %3 = test.shl %2, %a;
        %4 = test.eq %3, %a;
        %5 = test.neq %3, %a;
        %6 = test.effects <{ known = #builtin.bool<false> }>;
        %7 = test.call_one_of %a, @callee, @other(%3) : extern("C") (i32) -> i32;
        %8 = test.low %w;
        %9 = test.high %w;
        %10 = test.join %8, %9;
        test.sink %10, %a, %7 : (i128, i32, i32);
        test.store %p, %7 : (ptr<i32>, i32);
        %11 = test.load %p;
        test.touch_local <{ local = #builtin.local_variable<0, i32> }>;
        "test.reads_locals_and_holds"() ({ }) : () -> ();
        %12 = test.check_range %s <{ umin = #builtin.int<1 : i8>, umax = #builtin.int<-56 : i8> }>;
        builtin.ret %11 : (i32);
    };
    builtin.function public extern("C") @never(%x: i32) -> never {
        test.ret : ();
    };
};"#,
    ),
    (
        "ub",
        r#"builtin.function public extern("C") @ub(%c: i1) -> i32 {
    cf.cond_br %c ^dead, ^live : (i1);
^dead:
    ub.unreachable;
^live:
    %0 = ub.poison <{ value = #ub.poison<i32> }>;
    builtin.ret %0 : (i32);
};"#,
    ),
];
