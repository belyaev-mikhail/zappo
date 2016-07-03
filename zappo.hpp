#ifndef ZAPPO_HPP
#define ZAPPO_HPP

#include <cstdint>
#include <vector>
#include <limits>
#include <cstring>
#include <iostream>
#include <string>

#include <type_traits>

#include <cstring>

#ifdef USE_TYPESIG
#include <typesig.hpp>
#endif // USE_TYPESIG

#ifdef USE_RTTI
#include <typeinfo>
#endif

namespace zappo {

namespace ce {

template<class T>
constexpr T pow(T v, size_t p) {
    return (!p)? T(1) : v * pow(v, p - 1);
}

template<class T>
constexpr T atoi(T) {
    return T(0);
}
template<class T, class ...Chars>
constexpr T atoi(T radix, char h, Chars... t) {
    auto pw = pow(radix, sizeof...(t));
    auto rest = atoi(radix, t...);

    return (h >= 'A' && h <= 'F')? (h - 'A' + 10) * pw + rest:
           (h >= 'a' && h <= 'f')? (h - 'a' + 10) * pw + rest :
                                   (h - '0') * pw + rest;
}

template<class T> constexpr T min() { return std::numeric_limits<T>::max(); }
template<class T> constexpr T min(T h) { return h; }
template<class T0, class T1, class ...Ts>
constexpr T0 min(T0 h0, T1 h1, Ts... t) {
     auto rest = min(h1, t...);
     return (h0 > rest)? rest : h0;
}

template<class T> constexpr T max() { return std::numeric_limits<T>::min(); }
template<class T> constexpr T max(T h) { return h; }
template<class T0, class T1, class ...Ts>
constexpr T0 max(T0 h0, T1 h1, Ts... t) {
     auto rest = max(h1, t...);
     return (h0 < rest)? rest : h0;
}

template<class T> constexpr T sum() { return T{}; }
template<class T> constexpr T sum(T h) { return h; }
template<class T0, class T1, class ...Ts>
constexpr T0 sum(T0 h0, T1 h1, Ts... t) {
     auto rest = sum(h1, t...);
     return h0 + rest;
}

}

template<class Type, Type value_> struct integral_constant;
template<class VType, VType... Elements>
struct value_sequence;
template<char ... Chars> struct string_constant;
template<class T> struct type_tag;
template<class ...Ts> struct type_sequence;

template<class T>
struct error_type {};

namespace impl {
template<class T> using eval_t = typename T::type;
template<class T> using eval_v = integral_constant<decltype(T::value), T::value>;
} // namespace impl

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~integral_constant~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template<int v> using int_constant = integral_constant<int, v>;
template<size_t v> using size_constant = integral_constant<size_t, v>;
template<bool v> using bool_constant = integral_constant<bool, v>;
template<char c> using char_constant = integral_constant<char, c>;
using true_type = bool_constant<true>;
using false_type = bool_constant<false>;

template<class Type, Type value_>
struct integral_constant {
    static constexpr Type value = value_;

    constexpr operator Type() { return value; }

    template<class T>
    integral_constant<T, T(value)> to() { return {}; }

    template<class T>
    integral_constant<T, T(value)> to(type_tag<T>) { return {}; }

#define DEF_BINARY(OP) \
    template<class T, T v> \
    integral_constant<decltype(value OP v), (value OP v)> operator OP (integral_constant<T, v>) { return {}; }

    DEF_BINARY(==)
    DEF_BINARY(!=)
    DEF_BINARY(>)
    DEF_BINARY(<)
    DEF_BINARY(>=)
    DEF_BINARY(<=)
    DEF_BINARY(+)
    DEF_BINARY(-)
    DEF_BINARY(*)
    DEF_BINARY(/)
    DEF_BINARY(%)
    DEF_BINARY(&)
    DEF_BINARY(|)
    DEF_BINARY(^)
    DEF_BINARY(&&)
    DEF_BINARY(||)
    DEF_BINARY(>>)
    DEF_BINARY(<<)

#undef DEF_BINARY

#define DEF_PREFIX(OP) \
    integral_constant<decltype(OP value), (OP value)> operator OP () { return {}; }

    DEF_PREFIX(+)
    DEF_PREFIX(-)
    DEF_PREFIX(~)
    DEF_PREFIX(!)

#undef DEF_PREFIX

    type_tag<Type> type() { return {}; }

    friend std::ostream& operator<<(std::ostream& ost, integral_constant) { return ost << value; }
};

static true_type true_ct;
static false_type false_ct;

template<bool C, class T, class F>
std::conditional_t<C, T, F> ite(bool_constant<C>, T, F) { return {}; }

template<class T>
type_tag<T> tag(T) { return {}; }

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~value_sequence~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template<class VType, VType... Elements>
struct value_sequence;

namespace vs_impl {

template<char ... Chars>
string_constant<Chars...> vs_to_sc(value_sequence<char, Chars...>) { return {}; }

template<size_t Ix, class VS> struct index;
template<class Elem, Elem H, Elem... T>
struct index<0, value_sequence<Elem, H, T...>> { static constexpr auto value = H; };
template<class Elem, size_t Ix, Elem H, Elem ...T>
struct index<Ix, value_sequence<Elem, H, T...>> { static constexpr auto value = index<Ix-1, value_sequence<Elem, T...>>::value; };
template<class Elem, size_t Ix>
struct index<Ix, value_sequence<Elem>> {};

template<class SS0, class SS1> struct append;
template<class Elem, Elem ...C0, Elem ...C1>
struct append<value_sequence<Elem, C0...>, value_sequence<Elem, C1...>> { using type = value_sequence<Elem, C0..., C1...>; };
template<class SS0, class SS1>
using append_t = impl::eval_t< append<SS0, SS1> >;

template<size_t Ix, class VS> struct drop;
template<class Elem, Elem... T>
struct drop<0, value_sequence<Elem, T...>> { using type = value_sequence<Elem, T...>; };
template<class Elem, Elem H, Elem... T>
struct drop<0, value_sequence<Elem, H, T...>> { using type = value_sequence<Elem, H, T...>; };
template<class Elem, size_t Ix, Elem H, Elem ...T>
struct drop<Ix, value_sequence<Elem, H, T...>> { using type = impl::eval_t< drop<Ix-1, value_sequence<Elem, T...>> >; };
template<class Elem, size_t Ix>
struct drop<Ix, value_sequence<Elem>> { using type = value_sequence<Elem>; };
template<class Elem>
struct drop<0, value_sequence<Elem>> { using type = value_sequence<Elem>; };
template<size_t Ix, class VS>
using drop_t = impl::eval_t< drop<Ix, VS> >;

template<size_t Ix, class VS> struct take;
template<class Elem>
struct take<0, value_sequence<Elem>> { using type = value_sequence<Elem>; };
template<class Elem, Elem... T>
struct take<0, value_sequence<Elem, T...>> { using type = value_sequence<Elem>; };
template<class Elem, Elem H, Elem... T>
struct take<0, value_sequence<Elem, H, T...>> { using type = value_sequence<Elem>; };
template<class Elem, size_t Ix, Elem H, Elem ...T>
struct take<Ix, value_sequence<Elem, H, T...>> {
    using type = append_t<
                    value_sequence<Elem, H>,
                    impl::eval_t< take<Ix-1, value_sequence<Elem, T...>> >
                 >;
};
template<class Elem, size_t Ix>
struct take<Ix, value_sequence<Elem>> { using type = value_sequence<Elem>; };
template<size_t Ix, class VS>
using take_t = impl::eval_t< take<Ix, VS> >;

template<class VS, class F> struct filter;
template<class F, class Elem, Elem H, Elem ...T>
struct filter<value_sequence<Elem, H, T...>, F> {
    using tail = impl::eval_t< filter<value_sequence<Elem, T...>, F> >;
    using type = std::conditional_t<
                    decltype(std::declval<F>()(integral_constant<Elem, H>{}))::value,
                    append_t< value_sequence<Elem, H>, tail >,
                    tail
                 >;
};
template<class F, class Elem>
struct filter<value_sequence<Elem>, F> {
    using type = value_sequence<Elem>;
};
template<class VS, class F>
using filter_t = impl::eval_t< filter<VS, F> >;

std::ostream& print(std::ostream& ost) { return ost; }
template<class H>
std::ostream& print(std::ostream& ost, H h) { return ost << h; }
template<class H0, class H1, class ...Ts>
std::ostream& print(std::ostream& ost, H0 h0, H1 h1, Ts... ts) {
    return print(ost << h0 << ", ", h1, ts...);
}

template<class VType, VType... Elements>
value_sequence<VType, Elements...> mk_vs(integral_constant<VType, Elements>...) { return {}; }

} // namespace vs_impl

template<class VType, VType... Elements>
struct value_sequence {
    size_constant<sizeof...(Elements)> size() { return {}; }
    template<class Size, Size Ix>
    impl::eval_v<vs_impl::index<size_t(Ix), value_sequence>> operator[](integral_constant<Size, Ix>) { return {}; }

    true_type operator==(value_sequence) { return {}; }
    template<VType ...Other>
    false_type operator==(value_sequence<VType, Other...>) { return {}; }

    false_type operator!=(value_sequence) { return {}; }
    template<VType ...Other>
    true_type operator!=(value_sequence<VType, Other...>) { return {}; }

    template<VType... Others>
    value_sequence<VType, Elements..., Others...> operator+(value_sequence<VType, Others...>) { return {}; }

    template<class Size, Size Ix>
    vs_impl::drop_t<size_t(Ix), value_sequence> drop( integral_constant<Size, Ix> ) { return {}; }

    template<class Size, Size Ix>
    vs_impl::take_t<size_t(Ix), value_sequence> take( integral_constant<Size, Ix> ) { return {}; }

    template<VType... Others>
    auto starts_with(value_sequence<VType, Others...> that) {
        return take(that.size()) == that;
    }

    template<class Size, Size From, Size To>
    auto subseq(integral_constant<Size, From> from, integral_constant<Size, To> to) { return drop(from).take(to-from+integral_constant<Size, 1>{}); }

    template<class F>
    auto map(F f) {
        return make_sequence( f(integral_constant<VType, Elements>{})... );
    }

    template<class F>
    auto filter(F f) -> vs_impl::filter_t<value_sequence, F> { return {}; }

    integral_constant<VType, ce::min<VType>(Elements...)> min() { return {}; }
    integral_constant<VType, ce::max<VType>(Elements...)> max() { return {}; }
    integral_constant<VType, ce::sum<VType>(Elements...)> sum() { return {}; }

    template<class F>
    auto partition(F f) { return make_sequence(tag(filter(f)), tag(filter([&](auto&& c){ return !f(c); }))); }

private:
    template<class = VType>
    auto sort_impl(true_type) {
        return value_sequence<VType>{};
    }
    template<class = VType>
    auto sort_impl(false_type) {
        auto pivot = (*this)[size_constant<0>{}];
        auto rest = subseq(size_constant<1>{}, size() - size_constant<1>{});
        auto bigger = rest.filter([&](auto&& c){ return c > pivot; });
        auto smaller = rest.filter([&](auto&& c){ return c < pivot; });
        auto mid = rest.filter([&](auto&& c){ return c == pivot; });

        return smaller.sort() + mid + value_sequence<VType, pivot>{} + bigger.sort();
    }
public:
    auto sort() {
        return sort_impl<VType>(size() == size_constant<0>{});
    }

    type_tag<VType> type() { return {}; }

    friend std::ostream& operator<<(std::ostream& ost, value_sequence) { return vs_impl::print(ost << "{", Elements...) << "}"; }
};

template<class T, T ...Vs>
auto make_sequence(integral_constant<T, Vs>...) -> value_sequence<T, Vs...> { return {}; }
template<class T, T From>
auto range(integral_constant<T, From> from, integral_constant<T, From>) {
    return make_sequence(from);
}
template<class T, T From, T To>
auto range(integral_constant<T, From> from, integral_constant<T, To> to) {
    static_assert(From < To, "incorrect range()");
    return make_sequence(from) + range(integral_constant<T, From+1>{}, to);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~string_constant~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template<char ... Chars>
struct string_constant: value_sequence<char, Chars...> {
    using base = value_sequence<char, Chars...>;

    base as_seq() { return {}; }

    template<char... Others>
    string_constant<Chars..., Others...> operator+(string_constant<Others...>) { return {}; }

    template<class Size, Size Ix>
    auto drop( integral_constant<Size, Ix> ix) { return vs_impl::vs_to_sc(base::drop(ix)); }

    template<class Size, Size From, Size To>
    auto substr( integral_constant<Size, From> from, integral_constant<Size, To> to) { return vs_impl::vs_to_sc(base::subseq(from, to)); }
    template<class F> auto filter(F f) { return vs_impl::vs_to_sc(base::filter(f)); }

    static auto data() { static const char data_[] { Chars..., '\0' }; return data_; }
    friend std::ostream& operator<<(std::ostream& ost, string_constant) { return ost << data(); }
};
template<class Char, Char... chars>
string_constant<chars...> operator"" _ct() { return {}; }
template<class Int, int Radix, char... chars>
auto atoi(string_constant<chars...>) -> integral_constant<Int, ce::atoi<Int>(Radix, chars...)> { return {}; }

template<class Int, class SS>
auto parse_integer(SS ss_) {
    auto ss = ss_.filter([](auto c){ return c != char_constant<'\''>{}; });
    return ite(
        ss.starts_with("0x"_ct) || ss.starts_with("0X"_ct),
        atoi<Int, 16>(ss.drop(size_constant<2>{})),
        ite(
            ss.starts_with("0"_ct),
            atoi<Int, 8>(ss.drop(size_constant<1>{})),
            atoi<Int, 10>(ss)
        )
    );
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<char... Digits> auto operator"" _ct() { return parse_integer<int>(string_constant<Digits...>{}); }
template<char... Digits> auto operator"" _size_ct() { return parse_integer<size_t>(string_constant<Digits...>{}); }

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~type_tag~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<class T>
struct type_tag_base {
    using tagged_type = T;

    template<class T0 = T,
        class = std::enable_if_t<not std::is_function<T0>::value>
    >
    size_constant<sizeof(T0)> sizeofBytes() { return {}; }
    template<class T0 = T,
        class = std::enable_if_t<std::is_function<T0>::value>
    >
    size_constant<0> sizeofBytes() { return {}; }

    auto sizeofBits() { return sizeofBytes() * 8_size_ct; }

#define DEF_STD_CONVERTER(NAME) \
    type_tag<std::NAME##_t<T>> NAME() { return {}; }

    DEF_STD_CONVERTER(add_cv)
    DEF_STD_CONVERTER(add_const)
    DEF_STD_CONVERTER(add_volatile)
    DEF_STD_CONVERTER(remove_cv)
    DEF_STD_CONVERTER(remove_const)
    DEF_STD_CONVERTER(remove_volatile)
    DEF_STD_CONVERTER(remove_reference)
    DEF_STD_CONVERTER(add_lvalue_reference)
    DEF_STD_CONVERTER(add_rvalue_reference)
    DEF_STD_CONVERTER(remove_pointer)
    DEF_STD_CONVERTER(add_pointer)
    DEF_STD_CONVERTER(remove_extent)
    DEF_STD_CONVERTER(remove_all_extents)
    DEF_STD_CONVERTER(decay)
#undef DEF_STD_CONVERTER
    template<class TT = T,
        class = std::enable_if_t< std::is_signed<TT>::value || std::is_unsigned<TT>::value >
    >
    type_tag< std::make_signed_t<TT> > make_signed() { return {}; }
    template<class TT = T,
        class = std::enable_if_t< !std::is_signed<TT>::value && !std::is_unsigned<TT>::value >
    >
    type_tag< error_type<TT> > make_signed() { return {}; }

    template<class TT = T, class = std::enable_if_t< std::is_signed<TT>::value || std::is_unsigned<TT>::value>>
    type_tag< std::make_unsigned_t<TT> > make_unsigned() { return {}; }
    template<class TT = T, class = std::enable_if_t< !std::is_signed<TT>::value && !std::is_unsigned<TT>::value >>
    type_tag< error_type<TT> > make_unsigned() { return {}; }

    template<class TT = T, class = std::enable_if_t< std::is_enum<TT>::value>>
    type_tag< std::underlying_type_t<TT> > underlying_type() { return {}; }
    template<class TT = T, class = std::enable_if_t< !std::is_enum<TT>::value>>
    type_tag< error_type<TT> > underlying_type() { return {}; }

#define DEF_STD_CHECKER(NAME) \
    bool_constant<std::NAME<T>::value> NAME() { return {}; }

    DEF_STD_CHECKER(is_void)
    DEF_STD_CHECKER(is_null_pointer)
    DEF_STD_CHECKER(is_integral)
    DEF_STD_CHECKER(is_floating_point)
    DEF_STD_CHECKER(is_array)
    DEF_STD_CHECKER(is_enum)
    DEF_STD_CHECKER(is_union)
    DEF_STD_CHECKER(is_class)
    DEF_STD_CHECKER(is_function)
    DEF_STD_CHECKER(is_pointer)
    DEF_STD_CHECKER(is_lvalue_reference)
    DEF_STD_CHECKER(is_rvalue_reference)
    DEF_STD_CHECKER(is_member_object_pointer)
    DEF_STD_CHECKER(is_member_function_pointer)
    DEF_STD_CHECKER(is_fundamental)
    DEF_STD_CHECKER(is_arithmetic)
    DEF_STD_CHECKER(is_scalar)
    DEF_STD_CHECKER(is_object)
    DEF_STD_CHECKER(is_compound)
    DEF_STD_CHECKER(is_reference)
    DEF_STD_CHECKER(is_member_pointer)
    DEF_STD_CHECKER(is_const)
    DEF_STD_CHECKER(is_volatile)
    DEF_STD_CHECKER(is_trivial)
    DEF_STD_CHECKER(is_trivially_copyable)
    DEF_STD_CHECKER(is_standard_layout)
    DEF_STD_CHECKER(is_pod)
    DEF_STD_CHECKER(is_literal_type)
    DEF_STD_CHECKER(is_empty)
    DEF_STD_CHECKER(is_polymorphic)
    DEF_STD_CHECKER(is_final)
    DEF_STD_CHECKER(is_abstract)
    DEF_STD_CHECKER(is_signed)
    DEF_STD_CHECKER(is_unsigned)
    DEF_STD_CHECKER(is_copy_constructible)
    DEF_STD_CHECKER(is_trivially_copy_constructible)
    DEF_STD_CHECKER(is_nothrow_copy_constructible)
    DEF_STD_CHECKER(is_move_constructible)
    DEF_STD_CHECKER(is_trivially_move_constructible)
    DEF_STD_CHECKER(is_nothrow_move_constructible)
    DEF_STD_CHECKER(is_copy_assignable)
    DEF_STD_CHECKER(is_trivially_copy_assignable)
    DEF_STD_CHECKER(is_nothrow_copy_assignable)
    DEF_STD_CHECKER(is_move_assignable)
    DEF_STD_CHECKER(is_trivially_move_assignable)
    DEF_STD_CHECKER(is_nothrow_move_assignable)
    DEF_STD_CHECKER(is_destructible)
    DEF_STD_CHECKER(is_trivially_destructible)
    DEF_STD_CHECKER(is_nothrow_destructible)
    DEF_STD_CHECKER(has_virtual_destructor)
#undef DEF_STD_CHECKER
    template<class U>
    bool_constant<std::is_base_of<T, U>::value> is_base_of(type_tag<U>) { return {}; }

    true_type operator==(type_tag<T>) { return {}; }
    template<class U> false_type operator==(type_tag<U>) { return {}; }
    false_type operator!=(type_tag<T>) { return {}; }
    template<class U> true_type operator!=(type_tag<U>) { return {}; }

    template<class T0 = T, class ...Args>
    type_tag<T0(Args...)> operator()(type_tag<Args>...) { return {}; }

    friend std::ostream& operator<<(std::ostream& ost, type_tag_base) {
#if defined(USE_TYPESIG)
        return ost << typesig::signature<T>();
#elif defined(USE_RTTI)
        return ost << typeid(T).name();
#else
        return ost << "type_tag";
#endif
    }
};

template<class T>
struct type_tag: type_tag_base<T> {};

static type_tag<bool> bool_;
static type_tag<char> char_;

static type_tag<signed char> schar_;
static type_tag<signed short> short_;
static type_tag<signed int> int_;
static type_tag<signed long> long_;
static type_tag<signed long long> longlong_;

static type_tag<unsigned char> uchar_;
static type_tag<unsigned short> ushort_;
static type_tag<unsigned int> uint_;
static type_tag<unsigned long> ulong_;
static type_tag<unsigned long long> ulonglong_;

static type_tag<std::size_t> size_t_;
static type_tag<std::ptrdiff_t> ptrdiff_t_;
static type_tag<std::intptr_t> int_ptr_t_;
static type_tag<std::intmax_t> intmax_t_;
static type_tag<std::uintmax_t> uintmax_t_;

static type_tag<std::int8_t>  int8_t_;
static type_tag<std::int16_t> int16_t_;
static type_tag<std::int32_t> int32_t_;
static type_tag<std::int64_t> int64_t_;

static type_tag<std::uint8_t>  uint8_t_;
static type_tag<std::uint16_t> uint16_t_;
static type_tag<std::uint32_t> uint32_t_;
static type_tag<std::uint64_t> uint64_t_;

static type_tag<double> double_;
static type_tag<float> float_;
static type_tag<long double> long_double_;

template<class T, T ...Ts>
static std::ostream& operator<<(std::ostream& ost, type_tag<value_sequence<T, Ts...>>) {
    return ost << value_sequence<T, Ts...>{};
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~type_sequence~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template<class... Elements>
struct type_sequence;

namespace ts_impl {

template<size_t Ix, class TS> struct index;
template<class H, class... T>
struct index<0, type_sequence<H, T...>> { using type = H; };
template<size_t Ix, class H, class ...T>
struct index<Ix, type_sequence<H, T...>> { using type = impl::eval_t< index<Ix-1, type_sequence<T...>> >; };
template<size_t Ix>
struct index<Ix, type_sequence<>> {};
template<size_t Ix, class TS> using index_t = impl::eval_t< index<Ix, TS> >;

template<class TS0, class TS1> struct append;
template<class ...TS0, class ...TS1>
struct append<type_sequence<TS0...>, type_sequence<TS1...>> { using type = type_sequence<TS0..., TS1...>; };
template<class SS0, class SS1>
using append_t = impl::eval_t< append<SS0, SS1> >;

template<size_t Ix, class TS> struct drop;
template<class... T>
struct drop<0, type_sequence<T...>> { using type = type_sequence<T...>; };
template<class H, class... T>
struct drop<0, type_sequence<H, T...>> { using type = type_sequence<H, T...>; };
template<size_t Ix, class H, class ...T>
struct drop<Ix, type_sequence<H, T...>> { using type = impl::eval_t< drop<Ix-1, type_sequence<T...>> >; };
template<size_t Ix>
struct drop<Ix, type_sequence<>> { using type = type_sequence<>; };
template<>
struct drop<0, type_sequence<>> { using type = type_sequence<>; };
template<size_t Ix, class VS>
using drop_t = impl::eval_t< drop<Ix, VS> >;

template<size_t Ix, class TS> struct take;
template<>
struct take<0, type_sequence<>> { using type = type_sequence<>; };
template<class... T>
struct take<0, type_sequence<T...>> { using type = type_sequence<>; };
template<class H, class... T>
struct take<0, type_sequence<H, T...>> { using type = type_sequence<>; };
template<size_t Ix, class H, class ...T>
struct take<Ix, type_sequence<H, T...>> {
    using type = append_t<
                    type_sequence<H>,
                    impl::eval_t< take<Ix-1, type_sequence<T...>> >
                 >;
};
template<size_t Ix>
struct take<Ix, type_sequence<>> { using type = type_sequence<>; };
template<size_t Ix, class VS>
using take_t = impl::eval_t< take<Ix, VS> >;

template<class TS, class F> struct filter;
template<class F, class H, class ...T>
struct filter<type_sequence<H, T...>, F> {
    using tail = impl::eval_t< filter<type_sequence<T...>, F> >;
    using type = std::conditional_t<
                    decltype(std::declval<F>()(type_tag<H>{}))::value,
                    append_t< type_sequence<H>, tail >,
                    tail
                 >;
};
template<class F>
struct filter<type_sequence<>, F> {
    using type = type_sequence<>;
};
template<class VS, class F>
using filter_t = impl::eval_t< filter<VS, F> >;

template<class... Elements>
type_sequence<Elements...> mk_ts(type_tag<Elements>...) { return {}; }

std::ostream& print(std::ostream& ost) { return ost; }
template<class H>
std::ostream& print(std::ostream& ost, H h) { return ost << h; }
template<class H0, class H1, class ...Ts>
std::ostream& print(std::ostream& ost, H0 h0, H1 h1, Ts... ts) {
    return print(ost << h0 << ", ", h1, ts...);
}

} // namespace ts_impl

template<class... Elements>
struct type_sequence {
    size_constant<sizeof...(Elements)> size() { return {}; }
    template<class Size, Size Ix>
    ts_impl::index_t<size_t(Ix), type_sequence> operator[](integral_constant<Size, Ix>) { return {}; }

    true_type operator==(type_sequence) { return {}; }
    template<class... Other>
    false_type operator==(type_sequence<Other...>) { return {}; }

    false_type operator!=(type_sequence) { return {}; }
    template<class... Other>
    true_type operator!=(type_sequence<Other...>) { return {}; }

    template<class... Others>
    type_sequence<Elements..., Others...> operator+(type_sequence<Others...>) { return {}; }

    template<class Size, Size Ix>
    ts_impl::drop_t<size_t(Ix), type_sequence> drop( integral_constant<Size, Ix> ) { return {}; }

    template<class Size, Size Ix>
    ts_impl::take_t<size_t(Ix), type_sequence> take( integral_constant<Size, Ix> ) { return {}; }

    template<class... Others>
    auto starts_with(type_sequence<Others...> that) {
        return take(that.size()) == that;
    }

    template<class Size, Size From, Size To>
    auto subseq(integral_constant<Size, From> from, integral_constant<Size, To> to) {
        return drop(from).take(to-from+integral_constant<Size, 1>{});
    }

    template<class F>
    auto map(F f) {
        return make_sequence( f(type_tag<Elements>{})... );
    }

    template<class F>
    auto filter(F f) -> ts_impl::filter_t<type_sequence, F> { return {}; }

    friend std::ostream& operator<<(std::ostream& ost, type_sequence) { return ts_impl::print(ost << "{", type_tag<Elements>{}...) << "}"; }
};

template<class ...Vs>
auto make_sequence(type_tag<Vs>...) -> type_sequence<Vs...> { return {}; }

template<class ...Ts>
static std::ostream& operator<<(std::ostream& ost, type_tag<type_sequence<Ts...>>) {
    return ost << type_sequence<Ts...>{};
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

} // namespace zappo

#endif // ZAPPO_HPP
