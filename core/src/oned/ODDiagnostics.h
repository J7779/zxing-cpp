/*
* Copyright 2025 ZXing contributors
*/
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <functional>

namespace ZXing::OneD {

/**
 * Thread-local diagnostics collector for 1D barcode decode attempts.
 * When enabled (via Enable()), all 1D reader activity is logged with
 * high granularity so that the caller (e.g. JNI bridge) can extract
 * and forward the logs to the application layer.
 *
 * Usage:
 *   Diagnostics::Enable();
 *   ... call ReadBarcodes() ...
 *   auto logs = Diagnostics::Collect();
 *   Diagnostics::Disable();
 */
class Diagnostics
{
public:
	static void Enable()  { enabled() = true; lines().clear(); }
	static void Disable() { enabled() = false; lines().clear(); }
	static bool IsEnabled() { return enabled(); }

	static void Log(const std::string& msg)
	{
		if (enabled())
			lines().push_back(msg);
	}

	static void Log(const char* msg)
	{
		if (enabled())
			lines().emplace_back(msg);
	}

	/// Collect all accumulated log lines and clear the buffer.
	static std::vector<std::string> Collect()
	{
		auto result = std::move(lines());
		lines().clear();
		return result;
	}

	/// Get a reference to the current log lines without clearing.
	static const std::vector<std::string>& Peek()
	{
		return lines();
	}

private:
	static bool& enabled()
	{
		thread_local bool e = false;
		return e;
	}
	static std::vector<std::string>& lines()
	{
		thread_local std::vector<std::string> l;
		return l;
	}
};

// Convenience macro — only evaluates the expression when diagnostics are on
#define OD_DIAG(expr) do { if (::ZXing::OneD::Diagnostics::IsEnabled()) { ::ZXing::OneD::Diagnostics::Log(expr); } } while(0)

} // namespace ZXing::OneD
