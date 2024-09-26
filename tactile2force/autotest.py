import m0
import m1
import m2
import m2l
import m3
import m4
import m4_xela
import m5
import m5_light
import m6
import m7
import m8

fingers = ['index', 'middle', 'ring']
for finger in fingers:
    for i in range(8):
        n_train = 4
        n_test = 2
        patch_type = 'tip'
        m0.run_m0(n_train, n_test, finger, patch_type)
        m1.run_m1(n_train, n_test, finger, patch_type)
        m2.run_m2(n_train, n_test, finger, patch_type)
        m2l.run_m2l(n_train, n_test, finger, patch_type)
        m3.run_m3(n_train, n_test, finger, patch_type)
        m4.run_m4(n_train, n_test, finger, patch_type)
        m4_xela.run_m4x(n_train, n_test, finger, patch_type)
        m5.run_m5(n_train, n_test, finger, patch_type)
        m6.run_m6(n_train, n_test, finger, patch_type)
        m5_light.run_m5l(n_train, n_test, finger, patch_type)
        m7.run_m7(n_train, n_test, finger, patch_type)
        m8.run_m8(n_train, n_test, finger, patch_type)
        
    if finger == 'middle':
        patch_type = 'phal'
        for i in range(8):
            n_train = 2
            n_test = 2
            m0.run_m0(n_train, n_test, finger, patch_type)
            m1.run_m1(n_train, n_test, finger, patch_type)
            m2.run_m2(n_train, n_test, finger, patch_type)
            m2l.run_m2l(n_train, n_test, finger, patch_type)
            m3.run_m3(n_train, n_test, finger, patch_type)
            m4.run_m4(n_train, n_test, finger, patch_type)
            m4_xela.run_m4x(n_train, n_test, finger, patch_type)
            m5.run_m5(n_train, n_test, finger, patch_type)
            m6.run_m6(n_train, n_test, finger, patch_type)
            m5_light.run_m5l(n_train, n_test, finger, patch_type)
            m7.run_m7(n_train, n_test, finger, patch_type)
            m8.run_m8(n_train, n_test, finger, patch_type)
