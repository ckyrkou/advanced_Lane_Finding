#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 17:57:53 2017

@author: ckyrkou
"""

import pickle
import numpy as np
import math
import time

#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import os
import cv2

import argparse

# Define conversions in x and y from pixels space to meters
global ym_per_pix,xm_per_pix
global search_frames


def imshow(x,p=0):
    """
    Helper function to show image and pause program execution
    """
    cv2.imshow("Video",x)
    cv2.waitKey(p)

def size(x):
    """
    Helper function to get array size
    """
    return np.shape(x)

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """
    Do the sobel processing for either x or y mask
    Apply threshold and return binary image
    """
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    grad_binary = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    """
    Do the sobel processing and find the magnitude of the gradient
    Apply threshold and return binary image
    """
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1


    return mag_binary

def channel_thresh(im, thresh=(90,255)):
    """
    Apply threshold to a color channel with the input values and return binary image
    """
    binary = np.zeros_like(im)
    binary[(im > thresh[0]) & (im <= thresh[1])] = 1
    return binary
    
def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Do the sobel processing and find the direction of the gradient
    Apply threshold and return binary image
    """
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return dir_binary
    return dir_binary

def region_of_interest(img, vertices, t=255):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
  
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (t,) * channel_count
    else:
        ignore_mask_color = t
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    
    return masked_image
 
def createMaskVertices(image_size,v_top_side=0.35,v_bottom_side=0.1,v_top=0.5,v_bottom_offset=0,v_side_offset=0):
    """
    Finds the points of the trapezoid-like region of interest based on percentages of the input image dimensions
    Also can apply shifts and translations to adjust mask
    """
    v1=(int(image_size[1]*v_bottom_side-v_side_offset), (image_size[0]-1-v_bottom_offset))
    v2=(int(image_size[1]*v_top_side-v_side_offset), int(image_size[0]*v_top))
    v3=(int(image_size[1]*(1-v_top_side)-v_side_offset), int(image_size[0]*v_top))
    v4=(int(image_size[1]*(1-v_bottom_side)-v_side_offset), (image_size[0]-1-v_bottom_offset))
    vertices = np.array([[v1,v2,v3,v4]], dtype=np.int32)
    
    return vertices

def process_image(img,vertices):
    """
    Implements the main processing pipeline to find the binary image on which 
    the lines will be detected
    """
    #Blur the input image
    medBlur = cv2.GaussianBlur(img,(7,7),0)
    #Convert to HLS color space
    hls = cv2.cvtColor(medBlur,cv2.COLOR_BGR2HLS)
    #isolate saturation component
    S=hls[:,:,2]
    
    #COnvert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements
    
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    #output X direction gradient
    #imshow(scale_to_im(gradx))
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    #output Y direction gradient
    #imshow(scale_to_im(grady))
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(30,100))
    #output gradient magnitude
    #imshow(scale_to_im(mag_binary))
    dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(0.7, 1.3))
    #output gradient direction
    #imshow(scale_to_im(dir_binary))
    s_binary = channel_thresh(S, thresh=(100,255))
    #output Saturation color channel
    #imshow(scale_to_im(s_binary))
    
    combined = np.zeros_like(dir_binary)
    #cobine all the binary images from the different processing steps
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | ((s_binary == 1))] = 1
    
    # Apply masking to isolate region of interest             
    roi = region_of_interest(combined, vertices,t=1)
    
    img_out = np.zeros_like(img)   
    
    img_out[(roi == 1)] = img[(roi == 1)]
            
    return img_out,roi,combined

def transform_prespective(image,src,perc):
    """
    Changes the prespective of the in put image to match the src and dst points
    Dst points are based on an offset percentage of the image dimensions
    This function returns the warped image, and the matrix to inverse the transformation
    Also it returns the src and dst ponts to apply the inverse transformation
    """

    offset = image.shape[0]*perc
    # Grab the image shape
    img_size = (image.shape[1], image.shape[0])

    dst = np.float32([[offset,0], # v1
                      [offset,img_size[1]], # v0
                      [img_size[0]-offset,img_size[1]], # v3
                      [img_size[0]-offset,0]]) # v2

    M = cv2.getPerspectiveTransform(src, dst)

    Minv = cv2.getPerspectiveTransform(dst, src)
    
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    return warped,Minv,img_size[0]-offset,src,dst

def lane_histogram_detection(warped_image,left,right):
    """
    Performs the exhaustive line detection search by first performing histogram
    processing to find the possible line locations and then estimates the shape
    of the line using sliding window search. The parameters are the ones used in
    the lessons
    """
    
    out_img = np.dstack((warped_image, warped_image, warped_image))
    lines = np.dstack((warped_image, warped_image, warped_image))

    histogram = np.sum(warped_image[int(warped_image.shape[0]/2):,:], axis=0)
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(warped_image.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_image.shape[0] - (window+1)*window_height
        win_y_high = warped_image.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    ploty = np.linspace(0, warped_image.shape[0]-1, warped_image.shape[0] )
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    #Update line objects accordingly and draw lines    

    if(len(leftx)==0 or len(lefty)==0):
        left.detected = False        
    else:
        left.detected = True
        left_fit = np.polyfit(lefty, leftx, 2)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        lines[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        left = update_line_status(left,left_fit,left_fitx,leftx,lefty)
        
    if(len(rightx)==0 or len(righty)==0):
        right.detected = False        
    else:
        right.detected = True
        right_fit = np.polyfit(righty, rightx, 2)
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        lines[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        right = update_line_status(right,right_fit,right_fitx,rightx,righty)
    

    
    # Mark the base of each line
    left.line_base_pos = leftx_base
    left.line_base_pos = rightx_base

        
    return out_img,left,right,lines

def lane_histogram_skip_windows(binary_warped,left,right):
    """
    Having already calculated the line locations in the previous frame then it
    is much easier and faster to find the in this frame.
    """
    
    left_fit = left.best_fit
    right_fit = right.best_fit 
    
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    lines = np.dstack((binary_warped, binary_warped, binary_warped))
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    
    # Fit a second order polynomial to each
    if(len(leftx)==0 or len(lefty)==0):
        left.detected = False        
    else:
        left.detected = True
        left_fit = np.polyfit(lefty, leftx, 2)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        lines[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        left = update_line_status(left,left_fit,left_fitx,leftx,lefty)
        
    if(len(rightx)==0 or len(righty)==0):
        right.detected = False
    else:
        right.detected = True
        right_fit = np.polyfit(righty, rightx, 2)
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        lines[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        right = update_line_status(right,right_fit,right_fitx,rightx,righty)
       
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    return result,left,right,lines

def corrected_carviture(image,left,right):
    """
    Finds the curvature of the fitted polynomials and also correctst the meassurement
    for the real-world by multiplying the coefficients with ym_per_pix
    """
    ploty = np.linspace(0, image.shape[0]-1, num=image.shape[0])# to cover same y-range as image
    y_eval = np.max(ploty)
    
#   left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
#    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
                      
    left.radius_of_curvature = ((1 + (2*left.best_fit[0]*y_eval*ym_per_pix + left.best_fit[1])**2)**1.5) / np.absolute(2*left.best_fit[0])
    right.radius_of_curvature = ((1 + (2*right.best_fit[0]*y_eval*ym_per_pix + right.best_fit[1])**2)**1.5) / np.absolute(2*right.best_fit[0])
    # Now our radius of curvature is in meters
    
    # Example values: 632.1 m    626.2 m
    return left,right

def find_diastance_from_center(left,right,img_cols):
    """
    Estimates the distance from the center and the location of the vehicle by
    finding the difference between the image midpoint and the lane lines midpoint
    """
    distance_from_center = (((left.bestx[-1]+right.bestx[-1])/2)-np.int(img_cols/2))*xm_per_pix

    return distance_from_center

def update_line_status(line,line_fit,line_fitx,linex,liney):
    """
    Update the line object. The update process changes depending 
    on the state of lines that have been recorded.
    """
    if(line.current_n == 0 or line.total_n==1):
        line.current_fit = line_fit
        line.recent_fits =  line_fit
        line.best_fit = line_fit
        line.recent_xfitted = line_fitx
        line.bestx = line_fitx
        line.allx = linex
        line.ally = liney
        line.current_n += 1
    else:
        if(line.current_n < line.total_n):
            line.diffs = line.current_fit-line_fit
            line.current_fit = line_fit
            line.recent_fits = np.row_stack((line.recent_fits,line_fit))
            line.best_fit = np.mean(line.recent_fits,axis=0)
            line.recent_xfitted = np.column_stack((line.recent_xfitted,line_fitx))
            line.bestx = np.mean(line.recent_xfitted,axis=1)
            line.allx = linex
            line.ally = liney
            line.current_n += 1
    
        if(line.current_n >= line.total_n):
            line.diffs = line.current_fit-line_fit
            line.current_fit = line_fit
            line.recent_fits = np.row_stack((line.recent_fits[1::,:],line_fit))
            line.best_fit = np.mean(line.recent_fits,axis=0)
            line.recent_xfitted = np.column_stack((line.recent_xfitted[:,1::],line_fitx))
            line.bestx = np.mean(line.recent_xfitted,axis=1)
            line.allx = linex
            line.ally = liney
            line.current_n += 1
    
    return line

    
def channels3(x):
    #Stack grayscale images together to increase the color channels to 3
    return np.dstack((x,x,x))

def sidebyside(x,y):
    #Concatenate images side by side (horizontally)
    return np.concatenate((x,y),axis=1)

def upanddown(x,y):
    #Concatenate images vertically
    return np.concatenate((x,y),axis=0)

def scale_to_im(x,a=0,b=255):
    """
    Normalize the image data with Min-Max scaling to a range of [a b]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    # TODO: Implement Min-Max scaling for grayscale image data
    ma=(np.max(x))
    if(ma == 0):
        return x.astype(np.uint8)
    mi=(np.min(x))
    normalized_data = ((x-mi)/ma) # normalize [0-1]
    normalized_data = (normalized_data*b + a*(1-normalized_data)) #Scale values here
    return normalized_data.astype(np.uint8)
    
def pause():
    """
    Helper function to pause the program execution
    """
    cv2.waitKey()
    return

def drawLines(left,right,image,Minv,undist):
    """
    Transforms the detect lines to the real-world perspective and blends 
    the image with the detected lines image
    """

    left_fitx = left.bestx
    right_fitx = right.bestx

    ploty = np.linspace(0, image.shape[0]-1, num=image.shape[0])# to cover same y-range as image
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])

    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
        
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result

# Define a class to receive the characteristics of each line detection
class Line():
    """
    Define the Line class and all the necessary variales to track and record 
    recent and past lines
    """
    def __init__(self,n=10):
        # count of current fit
        self.current_n = 0
        # total number of previous fits to consider
        self.total_n = n
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients averaged over the last n iterations
        self.recent_fits = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        


def line_tracking_pipeline(image,mtx,dist,vertices,skip_count,left,right,src,perc):
    """
    This function implements the main processing pipeline. 
    """
    #Show input image
    #imshow(image)
    
    #1. Undistored the Image
    img = np.copy(cv2.undistort(image, mtx, dist, None, mtx))
    
    #Show undistorted image
    #imshow(img)
    
    #2. Extract lines 
    color_lines,binary_lines,_ = process_image(img,vertices)
    
    #Display Binary Image
    
    #Display Thresholded Region of Interest
    #imshow(binary_lines)
    
    #3. Isolate Region of Interest
    color_roi = region_of_interest(img, vertices)
    
    #Display Region of Interest
    #imshow(color_roi)
    
    #4. Change Image perspective
    warped_image_col,Minv,area,src,dst = transform_prespective(color_lines,src,perc)
    
    #Display Color thresholded Warped Image
    #imshow(warped_image_col)
    
    #Display Source and Dst warped points
    #imshow(cv2.addWeighted(img, 1, cv2.fillPoly(np.zeros_like(img), np.array([[src]],dtype=np.int32), (0,0,255)), 0.5, 0))
    #imshow(cv2.addWeighted(warped_image_col, 1, cv2.fillPoly(np.zeros_like(warped_image_col), np.array([[dst]],dtype=np.int32),(0,255,255)), 0.3, 0))
    
    warped_image,Minv,area,_,_ = transform_prespective(binary_lines,src,perc)
  
    
    #5. Detect lines
    if(skip_count == 0 or left.detected == False or right.detected == False):
        # Exhaustive Search 
        detected_lanes,left,right,lines = lane_histogram_detection(warped_image,left,right)
    else:
        # Fast Search
        detected_lanes,left,right,lines = lane_histogram_skip_windows(warped_image,left,right)
        if(left.detected == False or right.detected == False):
            skip_count=search_frames
    
    #6. Find Curvature
    left,right = corrected_carviture(image,left,right)
    
    #7.Find distance from center
    distance_from_center = find_diastance_from_center(left,right,img.shape[1])

    #imshow(detected_lanes)
    
    lines_drawn_img = drawLines(left,right,warped_image,Minv,img)
    
    if(distance_from_center > 0):
        side = 'Left'
    else:
        side = 'Right'
    
    #8. VISUALIZE!!!
        
    text = ('Distance from center: '+str(abs(round(distance_from_center,3)))+' m to the '+side)
    cv2.putText(lines_drawn_img,text,(10,50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,200,255),5)
    text = ('Average Curvature: '+str(round((left.radius_of_curvature+right.radius_of_curvature)/2,3))+' m')
    cv2.putText(lines_drawn_img,text,(10,150),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,200,255),5)
    
    newwarp = cv2.warpPerspective(scale_to_im(lines), Minv, (image.shape[1], image.shape[0])) 
    # Show corrected lines
    #imshow(newwarp)
    
    # Combine the result with the original image
    lines_drawn_img = cv2.addWeighted(lines_drawn_img, 0.85, newwarp, 1, 0)
    
    #Show line detection output
    #imshow(lines_drawn_img)
    
    result = sidebyside(cv2.resize(upanddown(upanddown(color_roi,scale_to_im(warped_image_col)),scale_to_im(detected_lanes)),(430,720)),lines_drawn_img)
    
    return left,right,result
        
#####---------------------------------------------------------------
## MAIN LOOP
#####---------------------------------------------------------------

##Load camera calibration parameters

p=open("cam_cal_params.pkl","rb")
calib_data = pickle.load(p)
mtx, dist = calib_data['mtx'], calib_data['dist']

##Read input
## Run python3 P4.py 0 to process video
## Run python3 P4.py 1 to process images

parser = argparse.ArgumentParser(description='Remote Driving')
parser.add_argument('doImage', type=int,
help='Process Images (1) or Video (0): python3 P4.py [number]')
args = parser.parse_args()
    
doImage=args.doImage

##Set src and offset for perspective transformation

src = np.float32([[578,464], # v1
                  [256,684], # v0
                  [1040,684], # v3
                  [707,464]]) # v2

perc =0.25

##Process the images in the test_images folder
if(doImage == 1):
       
    ind=0
    dir='./test_images/'
    ext=".jpg"
    
    for file in os.listdir(dir):
        if file.endswith(".jpg"):
            right = Line(n=10)
            left = Line(n=10)
            image=cv2.imread(dir+file)
            print("Reading: ",dir+file)
            ym_per_pix = 30/(image.shape[0])
            xm_per_pix = 3.7/(560)
            vertices = createMaskVertices((image.shape[0],image.shape[1]), v_top=0.63,v_top_side=.42,v_bottom_side=0.10,v_bottom_offset=0)
            left,right,img_out = line_tracking_pipeline(image,mtx,dist,vertices,0,left,right,src,perc)
            cv2.imwrite('output_images/test_'+str(ind)+'.jpg',img_out)
            ind+=1

# Do not process video if processed images - End Program
if(doImage == 1):
    exit()

#Record Video if necessary
record = 1
if(record == 1):
    filename = "./Capture_Video/"+time.strftime("%H:%M:%S")+".avi"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename,fourcc,25,(1710,720))  #size is (column,row)


#Video file to process
videofile = './project_video.mp4'
cap = cv2.VideoCapture(videofile)

cv2.namedWindow("Video",0)

#Initialize Variables
frame_id = 0
skip_count = 0
search_frames = 30

#Instantiate the line objects with a tracking history of 10
right = Line(n=10)
left = Line(n=10)


# Select a different region of interest for different videos
if(videofile[2] == 'p'):
    #ROI for project video
    vertices = createMaskVertices((720,1280), v_top=0.63,v_top_side=.43,v_bottom_side=0.10,v_side_offset=-20,v_bottom_offset=0)
else:
    #ROI for challenge video
    vertices = createMaskVertices((720,1280), v_top=0.63,v_top_side=.43,v_bottom_side=0.10,v_side_offset=-20,v_bottom_offset=0)


# Main loop for histogram detection and tracking
while(True):

    ret,image = cap.read()
    if(ret == False):	
        break
    else:
        # Initialize parameters on first frame
        if(frame_id == 0):
            ym_per_pix = 30/(image.shape[0])
            xm_per_pix = 3.7/(560)
        frame_id+=1
        if(frame_id < 0):
            continue
        
        # Initialize parameters on first frame
        # Do the main processing
        left,right,img_out = line_tracking_pipeline(image,mtx,dist,vertices,skip_count,left,right,src,perc)
        
        # Count number of frames to call the appropriate line detection function
        if(skip_count == search_frames):
            skip_count = 0
        else:
            skip_count+=1 
            
    imshow(img_out,p=1)
    
    k=cv2.waitKey(33) 

    if(record == 1):
        out.write(img_out)
    
    if(k & 0xFF ==ord('q')):
         break
    if(k & 0xFF ==ord('s')):
        cv2.imwrite('output_images/video_frame_'+str(frame_id)+'.jpg',img_out)
     
if(record == 1):
    out.release()

cap.release()
cv2.destroyAllWindows()
