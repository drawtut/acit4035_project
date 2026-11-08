"""Settings for the foot + pelvis FPA recorder and analysis.

Edit this file to match how the two WT901C485-E sensors are mounted.
Both fpa_pelvis_recorder.py and fpa_pelvis_analysis.py read it.

Sensor axes: look at the axis arrows printed on the sensor housing.
"""

# --- Which foot carries the foot sensor ------------------------------------
FOOT_SIDE = "right"            # "left" or "right"

# --- Forward axis of each sensor (the sensor axis that points forward) -----
# Foot sensor on the dorsum (top) of the foot, lying flat:
#   the printed axis that points towards the toes.
FOOT_FORWARD_AXIS = (0.0, 1.0, 0.0)      # +Y towards the toes

# Waist sensor on the sacrum (middle of the lower back, belt level),
# standing upright with its flat face against the body:
#   the axis that points FORWARD, i.e. into the body / towards the navel.
#   If the printed Z arrow points away from the body, forward is -Z.
PELVIS_FORWARD_AXIS = (0.0, 0.0, -1.0)   # -Z into the body

# Leave the sensors' installation direction (register 0x23) at HORIZONTAL,
# the factory default. The forward axes above handle any mounting.

# --- Sign of the result ------------------------------------------------------
# FPA is reported with toe-out positive and toe-in negative, for either foot.
# If a deliberate toe-out walk comes out negative, set this to -1.
FPA_SIGN = +1

# --- Protocol timing (seconds) -----------------------------------------------
STILL_SECONDS = 5.0           # stand still at the start and at the end

# --- Detection thresholds ------------------------------------------------------
STANCE_GYRO_DPS = 25.0        # foot |gyro| below this = foot on the ground
STANCE_ACCEL_TOL = 2.5        # and | |acc| - g | below this (m/s^2)
MIN_STANCE_S = 0.15           # shortest accepted stance
MAX_STANCE_S = 2.0            # longer than this = a pause, not a step
STILL_GYRO_DPS = 6.0          # both sensors below this = standing still
MIN_CAL_S = 2.0               # shortest accepted start/end still period
TURN_LIMIT_DEG = 20.0         # pelvis turning more than this within one
                              # stride = stride flagged as a turn
MIN_HORIZONTAL = 0.5          # forward axis must be at least this much
                              # horizontal (0-1), otherwise it is the wrong axis
