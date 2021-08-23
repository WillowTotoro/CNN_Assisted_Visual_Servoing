import numpy as np


def cal_robot_vel(intrin_para, bbcx, bbcy, dist, gain=1):
    
    cx = intrin_para[0]
    cy = intrin_para[1]
    fx = intrin_para[2]
    fy = intrin_para[3]

    if dist == 0:

        res = [0,0,0,0,0,0]
    else:

        bbcx = bbcx/608*1280
        bbcy = bbcy/608*720
        # img_jacobian = np.array(
        #     [[-fx/dist, bbcx/dist, -(fx+bbcx**2/fx)], [0, bbcy/dist, -bbcx*bbcy/fy]])
        # img_jacobian = np.array(
            # [[-fx/dist, 0, bbcx/dist, bbcx*bbcy/fx, -(fx+bbcx**2/fx),bbcy], [0,-fy/dist,bbcy/dist,fy+bbcy**2/fy,-bbcx*bbcy/fy,-bbcx]])
        img_jacobian = np.array(
            [[fx/dist, 0, bbcx/dist, -bbcx*bbcy/fx, (fx**2+bbcx**2)/fx,bbcy], [0, fy/dist, bbcy/dist,-(fy**2+bbcy**2)/fy,bbcx*bbcy/fy,-bbcx]])
        
        pixel_diff = np.array([[cx-bbcx], [cy-bbcy]])

        # pixel_diff = np.array([[bbcx-cx], [bbcy-cy]])

        inverse_jacobian = np.linalg.pinv(img_jacobian)

        vel_matrix = np.matmul(inverse_jacobian,gain*pixel_diff)
        
    
        # x = round(float(vel_matrix[2]),1)
        # z = round(float(vel_matrix[5]),1)
        # # z = round(float(vel_matrix[2]),1)
        # vel_matrix = [x,z]
        # output = []
        # print('original output:', vel_matrix)

        res = []

        for v in vel_matrix:
            if v[0] > 0:
                res.append(min(v[0],1.3))
            if v[0] < 0:
                res.append(max(v[0],-1.3))
        # print('velocity',res)

    # return ' '.join(str(v[0]) for v in vel_matrix)
    return res
