#pragma once
#include <ostream>
#include <sycl/sycl.hpp>

namespace syBlast
{
    /**
     * @namespace Complex 
     * 
     * @brief Stores SYCL compatible Complex Data Type
     * 
     * 
     */
    namespace Complex
    {

        template <typename T>
        class Complex
        {
        public:
            T _real;
            T _imag;
            constexpr Complex(T real = 0, T imag = 0) : _real(real), _imag(imag) {}

            constexpr T real() const { return this->_real; }
            constexpr T imag() const { return this->_imag; }
            constexpr void real(T r) { this->_real = r; }
            constexpr void imag(T i) { this->_imag = i; }

            constexpr Complex operator+() const { return Complex(this->_real, this->_imag); }
            constexpr Complex operator-() const { return Complex(-_real, -_imag); }

            constexpr Complex operator+(const Complex &other) const
            {
                return Complex(_real + other._real, _imag + other._imag);
            }

            constexpr Complex &operator+=(const Complex &other)
            {
                _real += other._real;
                _imag += other._imag;
                return *this;
            }

            constexpr Complex operator-(const Complex &other) const
            {
                return Complex(_real - other._real, _imag - other._imag);
            }

            constexpr Complex &operator-=(const Complex &other)
            {
                _real -= other._real;
                _imag -= other._imag;
                return *this;
            }

            constexpr Complex operator*(const Complex &other) const
            {
                return Complex(
                    _real * other._real - _imag * other._imag,
                    _real * other._imag + _imag * other._real);
            }

            constexpr Complex &operator*=(const Complex &other)
            {
                T new_real = _real * other._real - _imag * other._imag;
                T new_imag = _real * other._imag + _imag * other._real;
                _real = new_real;
                _imag = new_imag;
                return *this;
            }

            constexpr Complex operator/(const Complex &other) const
            {
                T denom = other._real * other._real + other._imag * other._imag;
                if (denom == 0)
                {
                    return Complex(0, 0);
                }
                return Complex((_real * other._real + _imag * other._imag) / denom,
                               (_imag * other._real - _real * other._imag) / denom);
            }

            constexpr Complex &operator/=(const Complex &other)
            {
                T denom = other._real * other._real + other._imag * other._imag;
                if (denom == 0)
                {
                    _real = 0;
                    _imag = 0;
                    return *this;
                }
                T new_real = (_real * other._real + _imag * other._imag) / denom;
                T new_imag = (_imag * other._real - _real * other._imag) / denom;
                _real = new_real;
                _imag = new_imag;
                return *this;
            }

            constexpr bool operator==(const Complex &other) const
            {
                return _real == other._real && _imag == other._imag;
            }

            constexpr bool operator!=(const Complex &other) const
            {
                return !(_real == other._real && _imag == other._imag);
            }

            constexpr Complex conjugate() const
            {
                return Complex(_real, -_imag);
            }

            constexpr Complex(const Complex &) = default;

            constexpr Complex &operator=(const Complex &) = default;

            ~Complex() = default;

            constexpr Complex operator+(const T &scalar) const
            {
                return Complex(_real + scalar, _imag);
            }

            constexpr Complex &operator+=(const T &scalar)
            {
                _real += scalar;
                return *this;
            }

            constexpr Complex operator-(const T &scalar) const
            {
                return Complex(_real - scalar, _imag);
            }

            constexpr Complex &operator-=(const T &scalar)
            {
                _real -= scalar;
                return *this;
            }

            constexpr Complex operator*(const T &scalar) const
            {
                return Complex(_real * scalar, _imag * scalar);
            }

            constexpr Complex &operator*=(const T &scalar)
            {
                _real *= scalar;
                _imag *= scalar;
                return *this;
            }

            constexpr Complex operator/(const T &scalar) const
            {
                if (scalar == 0)
                {
                    return Complex(0, 0);
                }
                return Complex(_real / scalar, _imag / scalar);
            }

            constexpr Complex &operator/=(const T &scalar)
            {
                if (scalar == 0)
                {
                    _real = 0;
                    _imag = 0;
                }
                else
                {
                    _real /= scalar;
                    _imag /= scalar;
                }
                return *this;
            }


        };


        template <typename T>
        constexpr Complex<T> operator+(const T &scalar, const Complex<T> &complex)
        {
            return Complex<T>(scalar + complex._real, complex._imag);
        }

        template <typename T>
        constexpr Complex<T> operator-(const T &scalar, const Complex<T> &complex)
        {
            return Complex<T>(scalar - complex._real, -complex._imag);
        }

        template <typename T>
        constexpr Complex<T> operator*(const T &scalar, const Complex<T> &complex)
        {
            return Complex<T>(scalar * complex._real, scalar * complex._imag);
        }

        template <typename T>
        constexpr Complex<T> operator/(const T &scalar, const Complex<T> &complex)
        {
            T denom = complex._real * complex._real + complex._imag * complex._imag;
            if (denom == 0)
            {
                return Complex<T>(0, 0);
            }
            return Complex<T>((scalar * complex._real) / denom,
                              (-scalar * complex._imag) / denom);
        }

        template <typename T>
        constexpr Complex<T> operator+(const Complex<T> &complex, const T &scalar)
        {
            return Complex<T>(scalar + complex._real, complex._imag);
        }

        template <typename T>
        constexpr Complex<T> operator-(const Complex<T> &complex, const T &scalar)
        {
            return Complex<T>(complex._real - scalar, -complex._imag);
        }

        template <typename T>
        constexpr Complex<T> operator*(const Complex<T> &complex, const T &scalar)
        {
            return Complex<T>(scalar * complex._real, scalar * complex._imag);
        }

        template <typename T>
        constexpr Complex<T> operator/(const Complex<T> &complex, const T &scalar)
        {
            return Complex<T>(complex._real / scalar, complex._imag / scalar);
        }

        template <typename T>
        std::ostream &operator<<(std::ostream &os, const Complex<T> &c)
        {
            os << "(" << c._real << "," << c._imag << ")";
            return os;
        }

        template <typename T>
        T absComplex(const Complex<T> &complex)
        {
            if constexpr (sizeof(T) <= sizeof(float))
            {
                return sycl::sqrt<float>(complex.real() * complex.real() + complex.imag() * complex.imag());
            }
            else
            {
                return sycl::sqrt<double>(complex.real() * complex.real() + complex.imag() * complex.imag());
            }
        }

        template <typename T>
        T argComplex(const Complex<T> &complex)
        {
            if constexpr (sizeof(T) <= sizeof(float))
            {
                return sycl::atan2<float>(complex.imag(), complex.real());
            }
            else
            {
                return sycl::atan2<double>(complex.imag(), complex.real());
            }
        }

        template <typename T>
        T normComplex(const Complex<T> &complex)
        {
            return complex.real() * complex.real() + complex.imag() * complex.imag();
        }

        template <typename T>
        Complex<T> conj(const Complex<T> &complex)
        {
            return complex.conjugate();
        }


    }
}