# Validation-fraction ablation

Sweep the within-fold direct-MLP training fraction while keeping the direct
predictor architecture, optimizer settings, adaptive-weight sharpness
(`kappa`) and Dyn3D margin fixed.  The complementary question subset is used
to estimate `v_avg`, preserving the main method's split protocol.
