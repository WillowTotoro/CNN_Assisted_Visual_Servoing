import numpy as np
from numpy.lib.function_base import diff
from numpy.linalg.linalg import inv


def cal_robot_vel_t(intrin_para, bbcx, bbcy, bbtlx,bbtly, dist, gain=1, gain_p=1):

    img_width = 1280
    img_height = 720

    yolo_img_size = 608
    desired_ratio = 0.4

    cx = intrin_para[0]
    cy = intrin_para[1]
    fx = intrin_para[2]
    fy = intrin_para[3]

    if dist == 0:
        res = [0,0,0,0,0,0]

    else:
        #convert yolo output according to image size

        yolo_cx = bbcx
        yolo_cy = bbcy
        yolo_tlx = bbtlx
        yolo_tly = bbtly

        bbcx = bbcx/yolo_img_size*img_width
        bbcy = bbcy/yolo_img_size*img_height
        bbtlx = bbtlx/yolo_img_size*img_width
        bbtly = bbtly/yolo_img_size*img_height

        #calculate image jacobian
        # img_jacobian = np.array(
            # [[fx/dist, 0, bbcx/dist, -bbcx*bbcy/fx, (fx**2+bbcx**2)/fx,bbcy], [0, fy/dist, bbcy/dist,-(fy**2+bbcy**2)/fy,bbcx*bbcy/fy,-bbcx]])
        
        img_jacobian_cen = np.array(
            [[fx/dist, 0, bbcx/dist, -bbcx*bbcy/fx, (fx**2+bbcx**2)/fx,bbcy], 
            [0, fy/dist, bbcy/dist,-(fy**2+bbcy**2)/fy,bbcx*bbcy/fy,-bbcx]])
        img_jacobian_area = np.array([[fx/dist, 0, bbtlx/dist, -bbtlx*bbtly/fx, (fx**2+bbtlx**2)/fx,bbtly], 
        [0,fy/dist,bbtly/dist,-(fy**2+bbtly**2)/fy,bbtlx*bbtly/fy,-bbtlx]])
            
        #calculate psedu inverse and transpose of image jacobian
        inverse_jacobian_cen = np.linalg.pinv(img_jacobian_cen)
        inverse_jacobian_area = np.linalg.pinv(img_jacobian_area)
        inverse_jacobian = np.concatenate((inverse_jacobian_cen,inverse_jacobian_area), axis=1)
        # trans_jacobian = np.transpose(img_jacobian)
        
        #calculate area error based on power function 
        area_func = np.array([[(bbcx-bbtlx)**2-(desired_ratio*img_width)**2/4],[(bbcy-bbtly)**2-(desired_ratio*img_height)**2/4]])
        # area_func = np.array([[(yolo_cx-yolo_tlx)**2-(desired_ratio*yolo_img_size)**2/4],[(yolo_cy-yolo_tly)**2-(desired_ratio*yolo_img_size)**2/4]])
        # area_func = np.array([[(desired_ratio*yolo_img_size)**2/4-(yolo_cx-yolo_tlx)**2],[(desired_ratio*yolo_img_size)**2/4-(yolo_cy-yolo_tly)**2]])
        
        # print('area_func', area_func)
        
        print('bb width:{}, bb height:{}'.format(2*(bbcx-bbtlx),2*(bbcx-bbtlx)))

        diff_area_func = np.array([[2*(bbcx-bbtlx)],[2*(bbcy-bbtly)]])
        bbw,bbh = diff_area_func
        # diff_area_func = np.array([[2*(yolo_cx-yolo_tlx)],[2*(yolo_cy-yolo_tly)]])
        # diff_area_func = np.array([[-2*(yolo_cx-yolo_tlx)],[-2*(yolo_cy-yolo_tly)]])
        # print('diff area function',diff_area_func)

        area_error = gain_p*np.array([min(0,area_func[0])*diff_area_func[0], min(0,area_func[1])*diff_area_func[1]])
        
        print('area error 1', area_error)

        # print((a1,a2))

        if bbw >= bbh:
            area_error = np.array([area_error[0], [0]])
        else:
            area_error = np.array([[0], area_error[1]])

        print('area error2', area_error)

        center_vel = np.array([[cx-bbcx], [cy-bbcy]])

        pixel_vel = np.concatenate((center_vel,area_error), axis=0)

        print('pixel vel', pixel_vel)

        vel_matrix = np.matmul(inverse_jacobian,gain*pixel_vel)
        
        # print('Bounding Box Area:', 4*abs(bbcx_yolo-x1)*abs(bbcy_yolo-y1))
        # print('Percentage Occupied:', 4*abs(bbcx-bbtlx)*abs(bbcy-bbtly)/(desired_ratio*img_height*img_width))
        
        res = []

        for v in vel_matrix:
            if v[0] > 0:
                res.append(min(v[0],1.3))
            if v[0] < 0:
                res.append(max(v[0],-1.3))
        print('velocity',res)
    
    return res


