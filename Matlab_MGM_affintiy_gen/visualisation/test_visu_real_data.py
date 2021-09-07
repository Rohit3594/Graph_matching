import os
import slam.io as sio
import slam.topology as stop
import slam.plot as splt
import slam.mapping as smap
import numpy as np

# plot with visbrain is cool but incompatible with old glibc lib on frioul :-(
#from vispy.scene import Line
#from visbrain.objects import VispyObj, SourceObj


db_name = 'OASIS'
fs_db_path = '/hpc/meca/data/OASIS/FS_OASIS'
input_data_dir = '/hpc/meca/data/OASIS/SulcalPits/OASIS_database/neuroImage_paper/OASIS_pits/subjects'

root_analysis_dir = '/hpc/meca/users/auzias/pits_graph_clustering/'+ db_name
experiment = 'oasis_pits02'

analysis_dir = os.path.join(root_analysis_dir, experiment)

param_string = 'dpfMap/alpha_0.03/an0_dn20_r1.5/alpha0.03_an0_dn20_r1.5'

if __name__ == "__main__":
    subjects_list = ['OAS1_0006']
    """ 'OAS1_0009', 'OAS1_0025', 'OAS1_0049', 'OAS1_0051', 'OAS1_0054', 'OAS1_0055',
                     'OAS1_0057', 'OAS1_0059', 'OAS1_0061', 'OAS1_0077', 'OAS1_0079', 'OAS1_0080', 'OAS1_0087',
                     'OAS1_0104', 'OAS1_0125', 'OAS1_0136', 'OAS1_0147', 'OAS1_0150', 'OAS1_0151', 'OAS1_0152',
                     'OAS1_0156', 'OAS1_0162', 'OAS1_0191', 'OAS1_0192', 'OAS1_0193', 'OAS1_0202', 'OAS1_0209',
                     'OAS1_0218', 'OAS1_0224', 'OAS1_0227', 'OAS1_0231', 'OAS1_0236', 'OAS1_0239', 'OAS1_0246',
                     'OAS1_0249', 'OAS1_0253', 'OAS1_0258', 'OAS1_0294', 'OAS1_0295', 'OAS1_0296', 'OAS1_0310',
                     'OAS1_0311', 'OAS1_0313', 'OAS1_0325', 'OAS1_0348', 'OAS1_0379', 'OAS1_0386', 'OAS1_0387',
                     'OAS1_0392', 'OAS1_0394', 'OAS1_0395', 'OAS1_0397', 'OAS1_0406', 'OAS1_0408', 'OAS1_0410',
                     'OAS1_0413', 'OAS1_0415', 'OAS1_0416', 'OAS1_0417', 'OAS1_0419', 'OAS1_0420', 'OAS1_0421',
                     'OAS1_0431', 'OAS1_0437', 'OAS1_0442', 'OAS1_0448', 'OAS1_0004', 'OAS1_0005', 'OAS1_0007',
                     'OAS1_0012', 'OAS1_0017', 'OAS1_0029', 'OAS1_0037', 'OAS1_0043', 'OAS1_0045', 'OAS1_0069',
                     'OAS1_0090', 'OAS1_0092', 'OAS1_0095', 'OAS1_0097', 'OAS1_0101', 'OAS1_0102', 'OAS1_0105',
                     'OAS1_0107', 'OAS1_0108', 'OAS1_0111', 'OAS1_0117', 'OAS1_0119', 'OAS1_0121', 'OAS1_0126',
                     'OAS1_0127', 'OAS1_0131', 'OAS1_0132', 'OAS1_0141', 'OAS1_0144', 'OAS1_0145', 'OAS1_0148',
                     'OAS1_0153', 'OAS1_0174', 'OAS1_0189', 'OAS1_0211', 'OAS1_0214', 'OAS1_0232', 'OAS1_0250',
                     'OAS1_0261', 'OAS1_0264', 'OAS1_0277', 'OAS1_0281', 'OAS1_0285', 'OAS1_0302', 'OAS1_0314',
                     'OAS1_0318', 'OAS1_0319', 'OAS1_0321', 'OAS1_0328', 'OAS1_0333', 'OAS1_0340', 'OAS1_0344',
                     'OAS1_0346', 'OAS1_0350', 'OAS1_0359', 'OAS1_0361', 'OAS1_0368', 'OAS1_0370', 'OAS1_0376',
                     'OAS1_0377', 'OAS1_0385', 'OAS1_0396', 'OAS1_0403', 'OAS1_0409', 'OAS1_0435', 'OAS1_0439',
                     'OAS1_0450']
    """
    hemi='l'


    BV_hem = hemi.upper()

    for subject in subjects_list:

        # read triangulated spherical mesh
        mesh_path = os.path.join(fs_db_path, subject, 'surf', '%sh.sphere.reg.gii' % hemi)
        sphere_mesh = sio.load_mesh(mesh_path)

        # get basins texture only to get the pole region (-1 in poles; everything 0 or above is a real basin with one pit)
        basins_path = os.path.join(input_data_dir, subject, '%s_%s_area50FilteredTexture.gii' % (param_string, BV_hem))
        basins_tex = sio.load_texture(basins_path)
        # create a binary texture with only the pole
        pole_tex = basins_tex.copy()
        basins_tex.darray[pole_tex.darray != -1] = 0
        # cut the mesh following the boundary of the pole texture
        sub_meshes, sub_tex, sub_corresp = stop.cut_mesh(sphere_mesh, basins_tex.darray[0])
        sub_meshes[0].show()
        # compute the boundary
        boundaries = stop.mesh_boundary(sub_meshes[0])

        planar_mesh = smap.disk_conformal_mapping(sub_meshes[0], boundary=boundaries[0])
        planar_mesh.show()


        # get pits texture (0 everywhere except single vertex with one where the pits are)
        pits_path = os.path.join(input_data_dir, subject, '%s_%s_area50FilteredTexturePits.gii' % (param_string, BV_hem))
        pits_tex = sio.load_texture(pits_path)
        pits_inds = np.where(pits_tex.darray[0] == 1)[0]


        # plot with visbrain is cool but incompatible with old glibc lib on frioul :-(
        # visb_sc = \
        #     splt.visbrain_plot(mesh=sphere_mesh, tex=basins_tex.darray[0],
        #                        caption='spherical mesh with pole')
        #
        # visb_sc = \
        #             splt.visbrain_plot(mesh=sub_meshes[0],
        #                                caption='open mesh', visb_sc=visb_sc)
        # ind = 0
        # cols = ['red', 'green', 'yellow', 'blue']
        # for bound in boundaries:
        #     s_rad = \
        #         SourceObj('rad', sub_meshes[0].vertices[bound],
        #                   color=cols[ind], symbol='square',
        #                   radius_min=10)
        #     visb_sc.add_to_subplot(s_rad)
        #     lines = Line(pos=sub_meshes[0].vertices[bound], color=cols[ind], width=10)
        #     # wrap the vispy object using visbrain
        #     l_obj = VispyObj('line', lines)
        #     visb_sc.add_to_subplot(l_obj)
        #     ind += 1
        #     if ind == len(cols):
        #         ind = 0
        #
        # visb_sc.preview()
