"""
Longitudinal Pair Creation Utilities.

Functions for creating baseline-followup pairs from medical imaging datasets,
based on temporal proximity and other criteria.
"""

import os
import shutil
from datetime import datetime
from typing import Optional, Tuple
from glob import glob

import pydicom
from tqdm import tqdm


def days_between(date1: str, date2: str, date_format: str = "%Y%m%d") -> int:
    """
    Calculate the number of days between two date strings.
    
    Args:
        date1: First date string
        date2: Second date string
        date_format: Format of the date strings (default: YYYYMMDD)
    
    Returns:
        Number of days between dates (can be negative)
    
    Example:
        >>> days_between("20230101", "20230115")
        14
    """
    d1 = datetime.strptime(date1, date_format)
    d2 = datetime.strptime(date2, date_format)
    return (d2 - d1).days


def is_within_n_days(date1: str, date2: str, n_days: int = 14) -> bool:
    """
    Check if two dates are within N days of each other.
    
    Args:
        date1: First date string (YYYYMMDD)
        date2: Second date string (YYYYMMDD)
        n_days: Maximum allowed days between dates
    
    Returns:
        True if dates are within n_days of each other
    """
    return abs(days_between(date1, date2)) <= n_days


def create_longitudinal_pairs(
    patient_dirs_pattern: str,
    max_days_apart: int = 14,
    view_position: str = 'AP',
    output_base: Optional[str] = None
) -> int:
    """
    Create longitudinal baseline-followup pair directories.
    
    Scans patient directories for studies that are temporally close
    and creates symlinked pair directories for training.
    
    Args:
        patient_dirs_pattern: Glob pattern for patient directories
        max_days_apart: Maximum days between baseline and followup
        view_position: Required view position for inclusion
        output_base: Base path for output. If None, creates in patient dir
    
    Returns:
        Number of pairs created
    
    Example:
        >>> count = create_longitudinal_pairs(
        ...     "/data/mimic/p*/p*",
        ...     max_days_apart=14,
        ...     view_position='AP'
        ... )
    """
    total_pairs = 0
    patient_dirs = glob(patient_dirs_pattern)
    
    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        # Get all series directories (exclude reports and existing pairs)
        series_dirs = [
            d for d in os.listdir(patient_dir)
            if not (d.endswith('.txt') or d.startswith('pair_'))
        ]
        
        if len(series_dirs) < 2:
            continue
        
        # Load study dates
        series_info = []
        for series in series_dirs:
            series_path = os.path.join(patient_dir, series)
            try:
                dcm_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
                if not dcm_files:
                    continue
                    
                dcm = pydicom.dcmread(os.path.join(series_path, dcm_files[0]))
                
                # Check view position
                if hasattr(dcm, 'ViewPosition') and dcm.ViewPosition != view_position:
                    continue
                
                series_info.append({
                    'path': series_path,
                    'name': series,
                    'date': dcm.StudyDate,
                    'time': getattr(dcm, 'StudyTime', '000000')
                })
            except Exception:
                continue
        
        # Sort by date and time
        series_info.sort(key=lambda x: (x['date'], x['time']))
        
        # Create pairs for consecutive studies within threshold
        for i in range(len(series_info) - 1):
            bl_info = series_info[i]
            fu_info = series_info[i + 1]
            
            if is_within_n_days(bl_info['date'], fu_info['date'], max_days_apart):
                _create_pair_directory(patient_dir, bl_info, fu_info)
                total_pairs += 1
    
    print(f"Created {total_pairs} longitudinal pairs")
    return total_pairs


def _create_pair_directory(
    patient_dir: str,
    baseline_info: dict,
    followup_info: dict
) -> str:
    """
    Create a pair directory with symlinks to baseline and followup studies.
    
    Args:
        patient_dir: Patient directory path
        baseline_info: Dict with 'path' and 'name' for baseline
        followup_info: Dict with 'path' and 'name' for followup
    
    Returns:
        Path to created pair directory
    """
    pair_name = f"pair_{baseline_info['name']}_{followup_info['name']}"
    pair_dir = os.path.join(patient_dir, pair_name)
    
    # Create baseline symlinks
    bl_dir = os.path.join(pair_dir, f"BL_{baseline_info['name']}")
    os.makedirs(bl_dir, exist_ok=True)
    for f in os.listdir(baseline_info['path']):
        src = os.path.join(baseline_info['path'], f)
        dst = os.path.join(bl_dir, f)
        if not os.path.lexists(dst):
            os.symlink(src, dst)
    
    # Create followup symlinks
    fu_dir = os.path.join(pair_dir, f"FU_{followup_info['name']}")
    os.makedirs(fu_dir, exist_ok=True)
    for f in os.listdir(followup_info['path']):
        src = os.path.join(followup_info['path'], f)
        dst = os.path.join(fu_dir, f)
        if not os.path.lexists(dst):
            os.symlink(src, dst)
    
    return pair_dir


def validate_pairs(
    pairs_pattern: str,
    max_days_apart: int = 14,
    view_position: str = 'AP',
    delete_invalid: bool = False
) -> Tuple[int, int]:
    """
    Validate existing longitudinal pairs.
    
    Checks that all pairs meet the temporal and view position criteria.
    
    Args:
        pairs_pattern: Glob pattern for pair directories
        max_days_apart: Maximum days between baseline and followup
        view_position: Required view position
        delete_invalid: Whether to delete invalid pair directories
    
    Returns:
        Tuple of (valid_count, invalid_count)
    """
    pair_dirs = glob(pairs_pattern)
    valid_count = 0
    invalid_count = 0
    
    for pair_dir in tqdm(pair_dirs, desc="Validating pairs"):
        try:
            # Find DICOM files
            bl_files = glob(os.path.join(pair_dir, 'BL_*', '*.dcm'))
            fu_files = glob(os.path.join(pair_dir, 'FU_*', '*.dcm'))
            
            if not bl_files or not fu_files:
                raise ValueError("Missing DICOM files")
            
            bl_dcm = pydicom.dcmread(bl_files[0])
            fu_dcm = pydicom.dcmread(fu_files[0])
            
            # Check view positions
            if bl_dcm.ViewPosition != view_position or fu_dcm.ViewPosition != view_position:
                raise ValueError(f"Invalid view position")
            
            # Check temporal proximity
            if not is_within_n_days(bl_dcm.StudyDate, fu_dcm.StudyDate, max_days_apart):
                raise ValueError(f"Studies too far apart")
            
            valid_count += 1
            
        except Exception as e:
            invalid_count += 1
            if delete_invalid:
                shutil.rmtree(pair_dir)
                print(f"Deleted invalid pair: {pair_dir} ({e})")
    
    print(f"Valid: {valid_count}, Invalid: {invalid_count}")
    return valid_count, invalid_count
