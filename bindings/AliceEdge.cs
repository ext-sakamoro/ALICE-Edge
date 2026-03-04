// SPDX-License-Identifier: MIT
// ALICE-Edge Unity C# Bindings — Auto-generated from ffi.rs
// Author: Moroya Sakamoto
//
// Usage:
//   1. Build alice-edge as cdylib: cargo build --release --features ffi
//   2. Copy libalice_edge.dylib / alice_edge.dll / libalice_edge.so to Assets/Plugins/
//   3. Add this file to your Unity project

using System;
using System.Runtime.InteropServices;

namespace AliceEdge
{
    [StructLayout(LayoutKind.Sequential)]
    public struct AliceLinearResult
    {
        public int slope;
        public int intercept;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct AliceQuadraticResult
    {
        public int a;
        public int b;
        public int c;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct AliceCubicResult
    {
        public int a;
        public int b;
        public int c;
        public int d;
    }

    public static class Native
    {
        private const string DllName = "alice_edge";

        // ── Linear fitting ──────────────────────────────────────────────

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern AliceLinearResult alice_fit_linear(int[] data, UIntPtr len);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int alice_evaluate_linear(int slope, int intercept, int x);

        // ── Quadratic fitting ───────────────────────────────────────────

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern AliceQuadraticResult alice_fit_quadratic(int[] data, UIntPtr len);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int alice_evaluate_quadratic(int a, int b, int c, int x);

        // ── Cubic fitting ───────────────────────────────────────────────

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern AliceCubicResult alice_fit_cubic(int[] data, UIntPtr len);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int alice_evaluate_cubic(int a, int b, int c, int d, int x);

        // ── Constant fitting ────────────────────────────────────────────

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int alice_fit_constant(int[] data, UIntPtr len);

        // ── Utility ─────────────────────────────────────────────────────

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int alice_int_to_q16(int i);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int alice_q16_to_int(int q);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern float alice_q16_to_f32(int q);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        public static extern bool alice_should_use_linear(int[] data, UIntPtr len);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern long alice_residual_error(int[] data, UIntPtr len,
                                                       int slope, int intercept);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr alice_edge_version();

        // ── Robust / SIMD ───────────────────────────────────────────────

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern AliceLinearResult alice_fit_linear_robust(int[] data, UIntPtr len, int madK);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern AliceLinearResult alice_fit_linear_simd(int[] data, UIntPtr len);

        // ── Filter / Delta ──────────────────────────────────────────────

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr alice_filter_outliers_mad(int[] data, UIntPtr len, int k,
                                                                [Out] int[] outBuf, UIntPtr outCapacity);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr alice_delta_encode(int[] pairs, UIntPtr numPairs);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr alice_delta_decode(int[] pairs, UIntPtr numPairs);

        // ── Security ────────────────────────────────────────────────────

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void alice_zeroize(int[] buf, UIntPtr len);
    }

    /// High-level wrapper for ALICE-Edge Q16.16 fixed-point model fitting
    public static class AliceModel
    {
        /// Fit a linear model to sensor data, returning (slope, intercept) in Q16.16
        public static (int slope, int intercept) FitLinear(int[] data)
        {
            var result = Native.alice_fit_linear(data, (UIntPtr)data.Length);
            return (result.slope, result.intercept);
        }

        /// Evaluate linear model at index x
        public static int EvaluateLinear(int slope, int intercept, int x)
        {
            return Native.alice_evaluate_linear(slope, intercept, x);
        }

        /// Convert Q16.16 to float
        public static float Q16ToFloat(int q)
        {
            return Native.alice_q16_to_f32(q);
        }

        /// Convert integer to Q16.16
        public static int IntToQ16(int i)
        {
            return Native.alice_int_to_q16(i);
        }

        /// Get library version string
        public static string Version()
        {
            IntPtr ptr = Native.alice_edge_version();
            return Marshal.PtrToStringAnsi(ptr);
        }
    }
}
