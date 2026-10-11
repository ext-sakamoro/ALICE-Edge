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

    /// Status codes returned by the robust / filter functions
    /// (see ALICE_EDGE_* in alice_edge.h)
    public static class AliceStatus
    {
        public const int Ok = 0;
        public const int NullPointer = -1;
        public const int NonPositiveK = -2;
        public const int Panic = -3;
    }

    /// A native call returned a status other than AliceStatus.Ok
    public class AliceEdgeException : Exception
    {
        public int Status { get; }

        public AliceEdgeException(int status, string message)
            : base(message)
        {
            Status = status;
        }
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

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr alice_edge_last_error();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void alice_edge_free_error_string(IntPtr s);

        // ── Robust / SIMD ───────────────────────────────────────────────

        /// k: outlier factor, a plain integer multiplier of the MAD (3 = three
        /// MADs, not Q16.16); k <= 0 returns AliceStatus.NonPositiveK
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int alice_fit_linear_robust(int[] data, UIntPtr len, int k,
                                                         out AliceLinearResult result);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern AliceLinearResult alice_fit_linear_simd(int[] data, UIntPtr len);

        // ── Filter / Delta ──────────────────────────────────────────────

        /// k: outlier factor, a plain integer multiplier of the MAD (3 = three
        /// MADs, not Q16.16); k <= 0 returns AliceStatus.NonPositiveK
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int alice_filter_outliers_mad(int[] data, UIntPtr len, int k,
                                                           [Out] int[] outBuf, UIntPtr outCapacity,
                                                           out UIntPtr outWritten);

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

        /// Robust linear fit (MAD rejection on the residuals), returning
        /// (slope, intercept) in Q16.16. k is a plain integer multiplier of the
        /// MAD (3 = three MADs, not Q16.16). Throws AliceEdgeException when
        /// k <= 0 (Status == AliceStatus.NonPositiveK) or on any other error.
        public static (int slope, int intercept) FitLinearRobust(int[] data, int k)
        {
            int status = Native.alice_fit_linear_robust(data, (UIntPtr)data.Length, k,
                                                        out AliceLinearResult result);
            ThrowIfError(status);
            return (result.slope, result.intercept);
        }

        /// Replace MAD outliers by the median. k is a plain integer multiplier
        /// of the MAD (3 = three MADs, not Q16.16). Throws AliceEdgeException when k <= 0 or on any other error.
        public static int[] FilterOutliersMad(int[] data, int k)
        {
            var outBuf = new int[data.Length];
            int status = Native.alice_filter_outliers_mad(data, (UIntPtr)data.Length, k,
                                                          outBuf, (UIntPtr)outBuf.Length,
                                                          out UIntPtr written);
            ThrowIfError(status);
            if ((ulong)written != (ulong)data.Length)
            {
                Array.Resize(ref outBuf, (int)written);
            }
            return outBuf;
        }

        private static void ThrowIfError(int status)
        {
            if (status == AliceStatus.Ok)
            {
                return;
            }
            string message = null;
            IntPtr ptr = Native.alice_edge_last_error();
            if (ptr != IntPtr.Zero)
            {
                message = Marshal.PtrToStringAnsi(ptr);
                Native.alice_edge_free_error_string(ptr);
            }
            throw new AliceEdgeException(status,
                message ?? $"alice_edge call failed with status {status}");
        }

        /// Get library version string
        public static string Version()
        {
            IntPtr ptr = Native.alice_edge_version();
            return Marshal.PtrToStringAnsi(ptr);
        }
    }
}
