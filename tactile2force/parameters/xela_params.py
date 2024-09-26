import numpy as np

patch_0 =  list(range(0, 4)) + list(range(4, 9)) + list(range(17, 23)) + list(range(31, 37)) + list(range(45, 50)) + list(range(58, 62))
patch_1A =  list(range(9, 13)) + list(range(23, 27)) + list(range(37, 41)) + list(range(50, 54))
patch_1B =  list(range(13, 17)) + list(range(27, 31)) + list(range(41, 45)) + list(range(54, 58))
patch_2 =  list(range(62, 66)) + list(range(66, 71)) + list(range(83, 89)) + list(range(101, 107)) + list(range(123, 128)) + list(range(144, 148))
patch_3A =  list(range(71, 75)) + list(range(89, 93)) + list(range(107, 111)) + list(range(128, 132))
patch_3B =  list(range(75, 79)) + list(range(93, 97)) + list(range(111, 115)) + list(range(132, 136))
patch_4A =  list(range(79, 83)) + list(range(97, 101)) + list(range(115, 119)) + list(range(136, 140))
patch_5 =  list(range(156, 160)) + list(range(164, 169)) + list(range(185, 191)) + list(range(203, 209)) + list(range(221, 226)) + list(range(246, 250))
patch_6A =  list(range(169, 173)) + list(range(191, 195)) + list(range(209, 213)) + list(range(226, 230))
patch_6B =  list(range(173, 177)) + list(range(195, 199)) + list(range(213, 217)) + list(range(230, 234))
patch_7A =  list(range(177, 181)) + list(range(199, 203)) + list(range(217, 221)) + list(range(234, 238))
patch_8 =  list(range(266, 270)) + list(range(278, 283)) + list(range(303, 309)) + list(range(329, 335)) + list(range(347, 352)) + list(range(364, 368))
patch_9A =  list(range(283, 287)) + list(range(309, 313)) + list(range(335, 339)) + list(range(352, 356))
patch_10A =  list(range(287, 291)) + list(range(313, 317)) + list(range(339, 343)) + list(range(356, 360))
patch_10B =  list(range(291, 295)) + list(range(317, 321)) + list(range(343, 347)) + list(range(360, 364))
patch_11 =  list(range(119, 123)) + list(range(140, 144)) + list(range(148, 152)) + list(range(152, 156)) + list(range(160, 164)) + list(range(181, 185))
patch_12 =  list(range(238, 242)) + list(range(250, 254)) + list(range(258, 262)) + list(range(270, 274)) + list(range(295, 299)) + list(range(321, 325))
patch_13 =  list(range(242, 246)) + list(range(254, 258)) + list(range(262, 266)) + list(range(274, 278)) + list(range(299, 303)) + list(range(325, 329))

patch = {
    '0':patch_0, 
    '1A':patch_1A, 
    '1B':patch_1B,
    '2':patch_2,
    '3A':patch_3A,
    '3B':patch_3B,
    '4A':patch_4A,
    '5':patch_5,
    '6A':patch_6A,
    '6B':patch_6B,
    '7A':patch_7A,
    '8':patch_8,
    '9A':patch_9A,
    '10A':patch_10A,
    '10B':patch_10B,
    '11':patch_11,
    '12':patch_12,
    '13':patch_13
    }

patch2tf = {
    '0':'/sensor_to_thumb',
    '2':'/sensor_to_index',
    '5':'/sensor_to_middle',
    '8':'/sensor_to_ring',
    '6A': '/sensor_to_middle_3rd_phal'
}

tip_patches = ['0', '2', '5', '8']
phalanges_patches = ['1A', '1B', '3A', '3B', '4A', '6A', '6B', '7A', '9A', '10A', '10B']
palm_patches = ['11', '12', '13']

index_finger = ['2', '3A', '3B', '4A']
middle_finger = ['5', '6A', '6B', '7A']
ring_finger = ['8', '9A', '10A', '10B']
thumb_finger = ['0', '1A', '1B']

theta_1 = 0.0
theta_2 = 1.5708
theta_3 =0.1745
theta_4 = 0.5236
theta_5 = 0.8727
theta_6 = -0.8378
theta_7 = 0.50618

class tip_tf():
    def __init__(self):
        self.rpy_dict = {
        "01": [0.0, 1.5708, 0.1745],
        "02": [0.0, 1.5708, 0.1745],
        "03": [0.0, 1.5708, 0.0],
        "04": [0.0, 1.5708, 0.0],
        "05": [0.0,1.5708, 0.5236],
        "06": [0.0, 0.8727, 0.1745],
        "07": [0.0, 0.8727,0.1745],
        "08": [0.0, 0.8727, 0.0],
        "09": [0.0, 0.8727, 0.0],
        "10": [-0.8378,0.50618,0.0],
        "11": [-0.5411, 0.4538, 0.0],
        "12": [-0.1745, 0.3840, 0.0],
        "13": [-0.1745, 0.3665, 0.0 ],
        "14": [0.0, 0.3142, 0.0],
        "15": [0.0, 0.3142, 0.0],
        "16": [-0.8378, -0.5062, 0.0 ],
        "17": [-0.5411, -0.4538, 0.0],
        "18": [ -0.1745, -0.3840, 0.0],
        "19": [-0.1745, -0.3665, 0.0],
        "20": [0.0, -0.3142, 0.0],
        "21": [0.0, -0.3142, 0.0],
        "22": [0.0, -1.5708, -0.5236],
        "23": [0.0, -0.8727, -0.1745],
        "24": [0.0, -0.8727, -0.1745],
        "25": [0.0, -0.8727, 0.0],
        "26": [0.0, -0.8727, 0.0],
        "27": [0.0, -1.5708, -0.1745],
        "28": [0.0, -1.5708, -0.1745],
        "29": [0.0, -1.5708, 0.0],
        "30": [0.0, -1.5708, 0.0]}

        self.measurement_to_taxel= [[0,-1,0],[-1,0,0],[0,0,-1]]
        self.sensor_o_to_fingertip= [[0,0,1],[1,0,0],[0,1,0]]


    def get_measurement_to_taxel(self):
        return np.array(self.measurement_to_taxel)

    def get_sensor_o_to_fingertip(self):
        return np.array(self.sensor_o_to_fingertip)

    def get_rpy_array(self):
        return np.array(list(self.rpy_dict.values()))
    
#tf = tip_tf()
#angles = list(tf.rpy_dict.values())
#angles = [item for sublist in angles for item in sublist]

N_TAXEL_TIP = 30
N_TAXEL_PHAL = 16
N_TAXEL_PALM = 24

phal_rot = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])