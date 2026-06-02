# Fight Detection Configuration Guide
# Campus Safety Video Surveillance System

## Overview

The enhanced fight detection system uses multi-factor analysis to distinguish
fighting behavior from similar activities like hugging, dancing, or sports.

## Detection Factors

| Factor | Description | Weight |
|--------|-------------|--------|
| Proximity | Distance between two persons < threshold | +2 |
| Overlap | Bounding boxes have significant IoU | +2 |
| Approaching | Persons moving toward each other | +1 |
| Chaos | Erratic movement patterns | +1 |
| Speed | Movement speed above threshold | +1 |

**Fight Score >= 3** required to trigger detection

## Configuration Parameters

### Fight Detection (fight)

```yaml
rules:
  fight:
    enabled: true
    distance_threshold: 150     # Max distance between centers (px)
    movement_threshold: 30        # Min speed to consider "moving" (px/s)
    min_duration_s: 0.5         # Min duration of sustained fight behavior
    debounce_s: 5.0             # Cooldown between fight alerts
    chaos_threshold: 100         # Motion chaos score threshold
    consecutive_frames: 3        # Frames of sustained behavior required
```

### Parameter Tuning Guide

#### For Indoor Campus Scenarios (Classrooms, Hallways)

```yaml
rules:
  fight:
    enabled: true
    distance_threshold: 120      # Smaller threshold for crowded spaces
    movement_threshold: 25       # Lower threshold for indoor movement
    min_duration_s: 0.8          # Longer duration to reduce false positives
    chaos_threshold: 80          # Higher chaos threshold
    consecutive_frames: 4
```

#### For Outdoor Campus Scenarios (Playgrounds, Sports Fields)

```yaml
rules:
  fight:
    enabled: true
    distance_threshold: 180      # Larger threshold for outdoor spaces
    movement_threshold: 35       # Higher threshold for active outdoor movement
    min_duration_s: 0.5
    chaos_threshold: 120
    consecutive_frames: 3
```

#### High Sensitivity (More Alerts, More False Positives)

```yaml
rules:
  fight:
    enabled: true
    distance_threshold: 200      # Very close detection range
    movement_threshold: 20       # Low speed threshold
    min_duration_s: 0.3         # Short duration
    chaos_threshold: 50          # Low chaos threshold
    consecutive_frames: 2
```

#### Low Sensitivity (Fewer Alerts, Fewer False Positives)

```yaml
rules:
  fight:
    enabled: true
    distance_threshold: 100      # Very close range only
    movement_threshold: 40       # High speed threshold
    min_duration_s: 1.0         # Long duration
    chaos_threshold: 150         # High chaos threshold
    consecutive_frames: 5
```

## Testing and Validation

### Test Scenarios

| Scenario | Expected Behavior | Adjustments |
|----------|-----------------|-------------|
| Two people fighting | HIGH confidence alert | Baseline |
| Two people hugging | No alert | Increase chaos_threshold |
| Two people dancing | No alert | Increase distance_threshold |
| Two people playing sports | Rare alert | Increase min_duration_s |
| One person running | No alert | Normal behavior |
| Group crowding | Crowd alert, not fight | Ensure crowd detection first |

### Test Video Specifications

```
Test Dataset Structure:
tests/
├── fight_detection/
│   ├── positive/           # Actual fighting scenarios
│   │   ├── scenario_001.mp4  # Two people physically fighting
│   │   ├── scenario_002.mp4  # Multiple person conflict
│   │   └── ...
│   ├── negative/           # Non-fighting similar scenarios
│   │   ├── hugging_001.mp4
│   │   ├── dancing_001.mp4
│   │   ├── sports_001.mp4
│   │   ├── walking_001.mp4
│   │   └── ...
│   └── expected_results.json
```

### Expected Performance Metrics

| Metric | Target | Minimum |
|--------|--------|---------|
| True Positive Rate | >= 95% | >= 85% |
| False Positive Rate | <= 5% | <= 15% |
| Detection Latency | <= 500ms | <= 1000ms |
| Precision | >= 90% | >= 80% |
| Recall | >= 85% | >= 75% |

## Common Issues and Solutions

### Issue: Too many false positives during sports activities

**Symptoms**: System alerts during basketball, soccer, or other sports

**Solutions**:
1. Increase `min_duration_s` to 1.0 or higher
2. Increase `chaos_threshold` to filter out regular sports movement
3. Enable crowd detection first to distinguish group activities
4. Use outdoor configuration preset

### Issue: Missed detections when people are partially occluded

**Symptoms**: Fighting not detected when people are close behind obstacles

**Solutions**:
1. Lower `distance_threshold` (but increases false positives)
2. Reduce `min_duration_s`
3. Ensure good camera coverage with minimal blind spots
4. Consider adding more cameras for overlapping coverage

### Issue: Different behavior patterns in different lighting

**Symptoms**: Detection works during day but fails at night

**Solutions**:
1. Use models trained on varied lighting conditions
2. Adjust confidence threshold based on time of day
3. Ensure adequate lighting in critical areas
4. Use IR cameras for low-light conditions

## Integration with Alert Systems

The fight detection can be integrated with:

- SMS alerts via Twilio
- Email notifications
- Push notifications via Firebase
- Campus security radio integration
- Automated video recording backup

## Model Improvement Recommendations

For significant accuracy improvements beyond parameter tuning:

1. **Fine-tune on campus-specific data**
   - Collect 5000+ images from actual campus environments
   - Include varied lighting, angles, and crowd densities
   - Annotate fighting and similar行为的边界情况

2. **Add pose estimation backbone**
   - Integrate OpenPose or similar for skeletal tracking
   - Analyze limb positions and angles
   - Detect specific fighting poses (punching, grappling)

3. **Add temporal modeling**
   - Use 3D convolutions or LSTM for sequence modeling
   - Analyze motion trajectories over 1-3 second windows
   - Distinguish chaotic fight motion from organized activity

4. **Add context awareness**
   - Time of day / location specific thresholds
   - Historical pattern analysis
   - Integration with access control events
