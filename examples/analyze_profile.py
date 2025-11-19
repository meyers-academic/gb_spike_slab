"""
Analyze JAX profiler output to identify bottlenecks.

Parses the trace.json.gz file and provides a summary of timing information.
"""

import json
import gzip
import os
from pathlib import Path
import numpy as np


def find_trace_file(base_dir="./jax_profiler_output"):
    """Find the most recent trace file."""
    profile_dir = Path(base_dir) / "plugins" / "profile"
    
    # Find most recent directory
    subdirs = [d for d in profile_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No profile directories found in {profile_dir}")
    
    latest_dir = max(subdirs, key=lambda x: x.stat().st_mtime)
    
    # Find trace.json.gz file
    trace_files = list(latest_dir.glob("*.trace.json.gz"))
    if not trace_files:
        raise FileNotFoundError(f"No trace.json.gz files found in {latest_dir}")
    
    return trace_files[0]


def parse_trace_file(trace_file):
    """Parse JAX trace JSON file."""
    print(f"Reading trace file: {trace_file}")
    
    with gzip.open(trace_file, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def analyze_trace_events(data):
    """Extract timing information from trace events."""
    # Check what keys are in the data
    print(f"Top-level keys in trace file: {list(data.keys())}")
    
    events = []
    if 'traceEvents' in data:
        events = data['traceEvents']
    elif isinstance(data, list):
        events = data
    else:
        print("Warning: Could not find traceEvents in expected format")
        return {}
    
    print(f"Total events found: {len(events)}")
    
    if len(events) > 0:
        print(f"Sample event keys: {list(events[0].keys())}")
        print(f"Sample event: {events[0]}")
        print()
    
    # Check time unit from metadata
    time_unit = data.get('displayTimeUnit', 'ns') if isinstance(data, dict) else 'ns'
    print(f"Time unit in trace file: {time_unit}")
    
    # JAX profiler uses nanoseconds
    # Convert to milliseconds
    def to_ms(dur_ns):
        return dur_ns / 1_000_000.0  # nanoseconds to milliseconds
    
    # Calculate actual wall-clock time span
    timestamps = []
    for event in events:
        if 'ts' in event:
            ts = event['ts']
            timestamps.append(ts)
            if 'dur' in event:
                timestamps.append(ts + event['dur'])
    
    wall_clock_ms = 0
    if timestamps:
        wall_clock_ms = to_ms(max(timestamps) - min(timestamps))
        print(f"Wall-clock time span: {wall_clock_ms:.2f} ms ({wall_clock_ms/1000:.2f} seconds)")
        print()
    
    # Extract all events with duration information
    event_times = {}
    events_by_category = {}
    
    for event in events:
        # Check for duration in different possible formats
        dur = None
        if 'dur' in event:
            dur = event['dur']  # nanoseconds in JAX trace format
        elif 'duration' in event:
            dur = event['duration']
        
        # Also check for phase-based events (begin/end pairs)
        ph = event.get('ph', '')
        name = event.get('name', event.get('cat', 'unknown'))
        cat = event.get('cat', 'unknown')
        
        # Track events by category
        if cat not in events_by_category:
            events_by_category[cat] = []
        events_by_category[cat].append(event)
        
        if dur is not None and dur > 0:
            # Convert nanoseconds to milliseconds
            dur_ms = to_ms(dur)
            
            if name not in event_times:
                event_times[name] = []
            event_times[name].append(dur_ms)
    
    # Print category summary
    print("Events by category:")
    for cat, cat_events in sorted(events_by_category.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {cat}: {len(cat_events)} events")
    print()
    
    # Calculate statistics
    summary = {}
    for name, times in event_times.items():
        if times:
            summary[name] = {
                'count': len(times),
                'total_ms': sum(times),
                'mean_ms': np.mean(times),
                'std_ms': np.std(times),
                'min_ms': min(times),
                'max_ms': max(times)
            }
    
    return summary, wall_clock_ms


def print_summary(summary):
    """Print a formatted summary of profiling results."""
    if not summary:
        print("No timing events found in trace file.")
        print("This might be because:")
        print("  1. The trace file format is different than expected")
        print("  2. Events are stored under different keys")
        print("  3. TensorBoard is the recommended way to view JAX profiles")
        return
    
    print("\n" + "=" * 80)
    print("Profiling Summary")
    print("=" * 80)
    print()
    
    # Filter out events with very small total time (likely rounding errors)
    # and sort by total time
    filtered_summary = {k: v for k, v in summary.items() if v['total_ms'] > 0.001}
    sorted_events = sorted(filtered_summary.items(), key=lambda x: x[1]['total_ms'], reverse=True)
    
    print(f"{'Event Name':<60} {'Count':<8} {'Total (ms)':<12} {'Mean (μs)':<12} {'Max (ms)':<12}")
    print("-" * 100)
    
    for name, stats in sorted_events[:30]:  # Top 30 events
        mean_us = stats['mean_ms'] * 1000  # Convert to microseconds
        print(f"{name[:58]:<60} {stats['count']:<8} {stats['total_ms']:<12.3f} "
              f"{mean_us:<12.2f} {stats['max_ms']:<12.3f}")
    
    print()
    print(f"Total events analyzed: {sum(s['count'] for s in summary.values())}")
    print(f"Events with measurable time (>0.001ms): {len(filtered_summary)}")
    print(f"Cumulative time (sum of all event durations): {sum(s['total_ms'] for s in summary.values()):.3f} ms")
    print("Note: This is cumulative time, not wall-clock time (events can overlap).")
    print("      If this is much less than wall-clock time, the profiler may not be")
    print("      capturing all execution. Try viewing in TensorBoard for full details.")
    print()


def main():
    print("=" * 80)
    print("JAX Profiler Output Analysis")
    print("=" * 80)
    print()
    
    try:
        trace_file = find_trace_file()
        data = parse_trace_file(trace_file)
        
        # Inspect structure
        print(f"Trace file structure:")
        if isinstance(data, dict):
            print(f"  Type: dict with keys: {list(data.keys())[:10]}")
        elif isinstance(data, list):
            print(f"  Type: list with {len(data)} items")
        print()
        
        summary, wall_clock_ms = analyze_trace_events(data)
        print_summary(summary)
        
        # Also try to extract XLA HLO information if available
        if isinstance(data, dict) and 'displayTimeUnit' in data:
            print(f"Time unit: {data['displayTimeUnit']}")
        print()
        
        print("Note: For detailed visualization, use TensorBoard:")
        print("  tensorboard --logdir=./jax_profiler_output")
        print()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you've run profile_waveforms.py first.")
    except Exception as e:
        print(f"Error parsing trace file: {e}")
        print("\nThe trace file format may be different. Try viewing in TensorBoard:")
        print("  tensorboard --logdir=./jax_profiler_output")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
