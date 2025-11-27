# Target-Localization-with-D435i-by-Color
Localization with D435i by color




# 🎨 HSV Color Calibration Guide

Different lighting conditions (sunny, cloudy, indoor lighting) significantly affect how a camera perceives color. To ensure the tracker works accurately in your specific environment, you need to calibrate the HSV thresholds using the provided tuning script.

### 1\. Setup

1.  Connect your RealSense D435i camera.
2.  Place the object you want to track (e.g., a green ball) in the camera's field of view.
3.  Run the calibration script:
    ```bash
    python3 hsv_tuner.py
    ```

### 2\. The Interface

Once the script is running, you will see four windows:

  * **`frame`**: The raw color video feed.
  * **`mask` (Most Important)**: A black-and-white view. **White** pixels represent the detected object, and **black** pixels represent the ignored background. **Your goal is to make your object pure white and the background pure black.**
  * **`result`**: The video feed showing *only* the detected color.
  * **`Trackbars`**: A control panel with 6 sliders to adjust the Lower (L) and Upper (U) bounds of Hue, Saturation, and Value.

### 3\. Calibration Steps

Follow this specific order to find the best values:

#### Step A: Initialize

Set the sliders to the following default positions to select **everything**:

  * **L - H**: 0  | **U - H**: 179
  * **L - S**: 0  | **U - S**: 255
  * **L - V**: 0  | **U - V**: 255
    *(The `mask` window should be completely white).*

#### Step B: Isolate Hue (Color Type)

1.  **Increase `L - H`**: Slowly drag the slider to the right until the object begins to turn black, then pull it back slightly until it becomes white again.
2.  **Decrease `U - H`**: Slowly drag the slider to the left until the object begins to disappear, then pull it back slightly.

<!-- end list -->

  * *Result:* Most background colors should disappear, leaving only objects with a similar color to your target.

#### Step C: Adjust Saturation (Color Purity)

1.  **Increase `L - S`**: Slowly drag to the right. This effectively filters out **white** or **pale** objects (like white walls, paper, or glare).

<!-- end list -->

  * *Result:* The mask should become cleaner, removing light background noise.

#### Step D: Adjust Value (Brightness)

1.  **Increase `L - V`**: Slowly drag to the right. This filters out **dark** regions and shadows.

<!-- end list -->

  * *Result:* The background should now be mostly black, with only your target object remaining white.

### 4\. Stability Check

Do not just calibrate for a stationary object.

1.  **Move the object** around the frame.
2.  Move it **closer and further** from the camera.
3.  Create some **shadows** with your hand.
4.  If the object disappears (turns black in the `mask`) during these movements, widen your thresholds slightly (decrease Lower values or increase Upper values).

### 5\. Update the Code

Once you are satisfied with the `mask`, write down the 6 numbers from the Trackbars. Update the `ColorTrackerNode` class in your `tracker_node.py` file:

```python
# Example: Replace these values with your calibration results
# Format: (Hue, Saturation, Value)

# Old values
# self.greenLower = (100, 43, 46)
# self.greenUpper = (124, 255, 255)

# YOUR NEW VALUES
self.greenLower = (35, 100, 100)  # [L-H, L-S, L-V]
self.greenUpper = (85, 255, 255)  # [U-H, U-S, U-V]
```

> **💡 Pro Tip for Red Objects:**
> In OpenCV, the Red Hue wraps around the 0-180 scale (it exists at both 0-10 and 170-180). If you are tracking red, you typically need to set the range to either `0-10` OR `170-179`, depending on which range covers your object better.
