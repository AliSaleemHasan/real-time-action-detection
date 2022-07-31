'''
This script implements a simple `Tracker` class.
'''


import numpy as np


class Tracker(object):
    ''' A simple tracker:

        For previous skeletons(S1) and current skeletons(S2),
        S1[i] and S2[j] are matched, if:
        1. For S1[i],   S2[j] is the most nearest skeleton in S2.
        2. For S2[j],   S1[i] is the most nearest skeleton in S1.
        3. The distance between S1[i] and S2[j] are smaller than self.dist_thresh.

        For unmatched skeletons in S2, they are considered
            as new people appeared in the video.
    '''

    def __init__(self, dist_thresh=0.1,score_thresh=0.5,num_of_keypoints=5, max_humans=6):
        '''
        Arguments:
            dist_thresh {float}: 0.0~1.0. The distance between the joints
                of the two matched people should be smaller than this.
                The image width and height has a unit length of 1.0.
            max_humans {int}: max humans to track.
                If the number of humans exceeds this threshold, the new
                skeletons will be abandoned instead of taken as new people.
        '''





        # threshold for skeleton keypoints score
        self.score_thresh = score_thresh

        # max number of keypoints to consider skeleton as person skeleton
        self.num_of_keypoints = num_of_keypoints


        # distanse threshold between kepoints
        self.dist_thresh = dist_thresh

        # max human that can be detected
        self.max_humans = max_humans

        # dictionary of id:skeleton
        self.id2skeleton = {}

        # variable that helps with putting ids to new people in image
        self.id = 0

        # dictionary of id:bounding_box
        self.id2box= {}

        # boolean array of people in frame
        # ex : if people = [False True False] then there is one person in image with id 1
        self.people = [False] * max_humans

    def track(self, curr_skels,curr_boundingBoxes,curr_people):
        ''' Track the input skeletons by matching them with previous skeletons,
            and then obtain their corresponding human id.
        Arguments:
            curr_skels {list of list}: each sub list is a person's skeleton.
        Returns:
            self.id2skeleton {dict}:  a dict mapping human id to his/her skeleton.
        '''


        # first we need to make sure to take just valid skeletons (which have more than keypoints with score > threshold)
        indeces = []

        for index,people in enumerate(curr_skels):

            counter = 0
            for keypoint in people:

                # keypoint[2]  is score of keypoint
                # if score of keypoint > score threshold then we will consider it
                if keypoint[2] >= self.score_thresh:
                    counter +=1

                # if we considered more than num_of_keypoints then the skeleton is valid
                if counter >=self.num_of_keypoints :

                    # save valid skeleton index
                    indeces.append(index)

                    # no need to do this for the rest of keypoints for this skeleton
                    break




        # take just valid skels
        curr_skels = curr_skels[indeces]

        # match features between skeleton and skeleton itself
        # because output of pose detection may be 2 skeletons for the same person



        # take bounding boxes of balid skels
        curr_boundingBoxes = curr_boundingBoxes[indeces]


        # number of valid skels 
        N = len(curr_skels)






        if len(self.id2skeleton) > 0:
            ids, prev_skels = map(list, zip(*self.id2skeleton.items()))

            # get matches between new and previous skeletons with number of current skeletons after remove duplicates
            good_matches , N = self._match_features(prev_skels, curr_skels)


            self.id2skeleton = {}
            self.id2box= {}
            self.people=[False] * self.max_humans

            # used to save matched skeleton between two frames
            is_matched = [False]*N

            # loob throw good matches to save them
            for i2, i1 in good_matches.items():
                human_id = ids[i1]
                self.id2skeleton[human_id] = np.array(curr_skels[i2])
                self.id2box[human_id]=np.array(curr_boundingBoxes[i2])
                self.people[i2]=True
                is_matched[i2] = True
            

            
            # get unmatched skeletons wich considered as new skeletons in new frame
            unmatched_idx = [i for i, matched in enumerate(
                is_matched) if not matched]


        else:
            good_matches = []
            unmatched_idx = range(N)
        


        # Add unmatched skeletons (which are new skeletons) to the list
        num_humans_to_add = min(len(unmatched_idx),
                                self.max_humans - len(good_matches))

        for i in range(num_humans_to_add):

            self.id = np.argmin(curr_people)
            self.people[self.id] = True
            self.id2skeleton[self.id] = np.array(
                curr_skels[unmatched_idx[i]])
            self.id2box[self.id] = np.array(
            curr_boundingBoxes[unmatched_idx[i]])

        return self.id2skeleton ,self.id2box, self.people


   
    def _match_features(self, features1, features2):
        ''' Match the features.　Output the matched indices.
        Returns:
            good_matches {dict}: a dict which matches the
                `index of features2` to `index of features1`.
        '''
        features1, features2 = np.array(features1), np.array(features2)


       
        

        features2 = self.actual_skels(features2)
        N = len(features2)
        # If f1i is matched to f2j and vice versa, the match is good.
        good_matches = {}
        n1, n2 = len(features1), len(features2)
        if n1 and n2:

            # dist_matrix[i][j] is the distance between features[i] and features[j]
            dist_matrix = [[self.cost(f1, f2) for f2 in features2]
                           for f1 in features1]
            dist_matrix = np.array(dist_matrix)

            # Find the match of features1[i]
            matches_f1_to_f2 = [dist_matrix[row, :].argmin()
                                for row in range(n1)]

            # Find the match of features2[i]
            matches_f2_to_f1 = [dist_matrix[:, col].argmin()
                                for col in range(n2)]

            for i1, i2 in enumerate(matches_f1_to_f2):
                if matches_f2_to_f1[i2] == i1 and dist_matrix[i1, i2] < self.dist_thresh:
                    good_matches[i2] = i1

            if 0:
                print("distance matrix:", dist_matrix)
                print("matches_f1_to_f2:", matches_f1_to_f2)
                print("matches_f1_to_f2:", matches_f2_to_f1)
                print("good_matches:", good_matches)

        return good_matches , N

    

    def cost(self,sk1, sk2):

        def calc_dist(p1, p2): return (
         (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

        joints = np.array([num for num in range(1,12)])


        sk1, sk2 = sk1[joints], sk2[joints]


        # take with score more than 0.05
        valid_idx = np.logical_and(sk1[:,2] >= self.score_thresh, sk2[:,2] >= self.score_thresh)



        sk1, sk2 = sk1[valid_idx], sk2[valid_idx]


        sum_dist, num_points = 0, int(len(sk1)/2)

        # if there is no points in skeleton then the cost will be 99999
        if num_points == 0:
            return 99999

        else:
            # compute distance between each pair of joint
            for i in range(num_points):
                idx = i * 2
                for j in range(2):
                    sum_dist += calc_dist(sk1[idx:idx+2,j], sk2[idx:idx+2,j])
            mean_dist = sum_dist / (num_points *2)
            mean_dist /= (1.0 + 0.05*num_points)  # more points, the better
            return mean_dist

      
                
        # remove duplicates from current skeleton 
    def actual_skels(self,skels):

        distance_dict = {}
        added = []
        indeces = []
        #TODO: merge all loops together 
        for id1, f1 in enumerate(skels):
            distance_dict[id1] = []
            item_cost = self.cost(f1,f1)
            if id1 in added:
                continue
            for id2,f2 in enumerate(skels):
                costt = self.cost(f1,f2)
                if  abs(item_cost - costt)  < 0.05:
                    added.append(id2)
                    distance_dict[id1].append({'id':id2,'cost':costt})
                


        for key,value in distance_dict.items():
            if(len(value) == 0):
                continue
            indeces.append(key)
        return skels[indeces]


