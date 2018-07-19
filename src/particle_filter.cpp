/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <map>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	default_random_engine gen;
	// This line creates a normal (Gaussian) distribution for x, y, theta.
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	//std::cout << "ParticleFilter::init: ";
	num_particles = 10;
	for(int i = 0; i < num_particles; ++i)
	{
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		//std::cout << p.x << "," << p.y << "|";
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		
		particles.push_back(p);
	}
	//std::cout << std::endl;

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

	//std::cout << "ParticleFilter::prediction: ";

	float vel_div_theta = 0.0;
	float delta_theta = yaw_rate * delta_t;
	if(fabs(yaw_rate) > 0.00001)
	{
		vel_div_theta = (velocity/yaw_rate);
	}

	for(int i = 0; i < num_particles; ++i)
	{
		float x = particles[i].x + vel_div_theta * (sin(particles[i].theta + delta_theta) - sin(particles[i].theta));
		float y = particles[i].y + vel_div_theta * (cos(particles[i].theta) - cos(delta_theta + particles[i].theta));
		float theta = particles[i].theta + delta_theta;
		
		// This line creates a normal (Gaussian) distribution for x, y, theta.
		normal_distribution<double> dist_x(x, std_pos[0]);
		normal_distribution<double> dist_y(y, std_pos[1]);
		normal_distribution<double> dist_theta(theta, std_pos[2]);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);

		//std::cout << particles[i].x << "," << particles[i].y << "|";
	}
	//std::cout << std::endl;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// For each particle:
	// For each observation:
	// 1. Transform: Observations to map coordinates.
	// 2. Associate: each observation to map landmarks
	// 3. Update weights:
	// 3.1. Determine measurement probabilities.
	// 3.2. Combine probabilities. (Posterior Probability).

	double sig_x = std_landmark[0];
	double sig_y = std_landmark[1];
	//double sig_theta = std_landmark[2];
	//std::cout << "ParticleFilter::updateWeights: " << sig_x << ", " << sig_y << std::endl;
	double total_weight = 0.0;

	for(int i = 0; i < num_particles; ++i)
	{
		double particle_weight = 1.0;//particles[i].weight;
		double x_part = particles[i].x;
		double y_part = particles[i].y;
		double theta = particles[i].theta;
		//std::cout << "Particle " << i << ": (" << x_part << "," << y_part << ")" << std::endl;

		std::vector<LandmarkObs> obs_in_map_coords;
		std::vector<ObservationToLandmark> all_distances;

		int obs_size = observations.size();
		for(int j = 0; j < obs_size; ++j)
		{
			// 1. Transform: Observations to map coordinates.
			double x_obs = observations[j].x;
			double y_obs = observations[j].y;
			// Homogeneus transformation
			double x_map = x_part + (cos(theta) * x_obs) - (sin(theta) * y_obs);
			double y_map = y_part + (sin(theta) * x_obs) + (cos(theta) * y_obs);

			obs_in_map_coords.push_back({-1, x_map, y_map});

			int lsize = map_landmarks.landmark_list.size();
			for (int l = 0; l < lsize; ++l)
			{
				double euclidean_distance = dist(x_map, y_map,
												map_landmarks.landmark_list[l].x_f, map_landmarks.landmark_list[l].y_f);

				if(euclidean_distance > sensor_range)
					// Discard this landmark as it is outside sensor range
					continue;

				//std::cout << "Distance from (" << x_map << ", " << y_map << ")->("
				//			<< map_landmarks.landmark_list[l].x_f << ", " << map_landmarks.landmark_list[l].y_f << ") = " << euclidean_distance << std::endl;
				all_distances.push_back({l, j, euclidean_distance});
			}
		}

		// Now we reorder the array.
		std::sort(all_distances.begin(), all_distances.end());

		/*
		std::cout << "Observations: ";
		for(unsigned int i = 0; i < obs_in_map_coords.size(); ++i)
		{
			std::cout << i << "(" << obs_in_map_coords[i].x << ", " << obs_in_map_coords[i].y << ") | ";
		}
		std::cout << std::endl;

		std::cout << "Landmarks: ";
		for(unsigned int i = 0; i < map_landmarks.landmark_list.size(); ++i)
		{
			std::cout << i << "(" << map_landmarks.landmark_list[i].x_f << ", " << map_landmarks.landmark_list[i].y_f << ") | ";
		}
		std::cout << std::endl;

		std::cout << "Observations to Landmarks: ";
		for(unsigned int i = 0; i < all_distances.size(); ++i)
		{
			std::cout << "(L: " << all_distances[i].landmark_id << ", O: " << all_distances[i].obs_id << ", D: " << all_distances[i].distance << ") ";
		}
		std::cout << std::endl;
		*/

		// Now we pick shortest distances. First come first serve, as we are sorted by minimum distance
		std::map<int, bool> landmark_assigned;
		int size = all_distances.size();
		for (int i = 0, obs_assigned = 0; i < size && obs_assigned < obs_size; ++i)
		{
			const ObservationToLandmark& o2l = all_distances[i];

			if (obs_in_map_coords[o2l.obs_id].id != -1 )//|| landmark_assigned[o2l.landmark_id])
				// Observation already matched with landmark
				continue;

			// Do the match
			obs_in_map_coords[o2l.obs_id].id = o2l.landmark_id;
			landmark_assigned[o2l.landmark_id] = true;

			// 3. Update weights:
			double x_obs = obs_in_map_coords[o2l.obs_id].x;
			double y_obs = obs_in_map_coords[o2l.obs_id].y;
			double mu_x = map_landmarks.landmark_list[o2l.landmark_id].x_f;
			double mu_y = map_landmarks.landmark_list[o2l.landmark_id].y_f;

			// calculate normalization term
			double gauss_norm = 1 / (2*M_PI*sig_x*sig_y);
			// calculate exponent
			double exponent = ( (x_obs-mu_x)*(x_obs-mu_x)/(2 * sig_x * sig_x) ) + ( (y_obs-mu_y)*(y_obs-mu_y)/(2 * sig_y * sig_y) );
			// calculate weight using normalization terms and exponent
			double weight = gauss_norm * exp(-exponent);

			/*
			std::cout << "Obs: " << o2l.obs_id << ": (" << x_obs << "," << y_obs << ")";
			std::cout << ", Landmark " << o2l.landmark_id << ": (" << mu_x << "," << mu_y << ")";
			std::cout << ", dist: " << o2l.distance;
			std::cout << ", gauss_norm: " << gauss_norm << ", exponent: " << exponent << ", exp: " << exp(-exponent) << ", weight: " << weight;
			std::cout << std::endl;
			*/

			if(weight > 0.0)
			{
				particle_weight *= weight;
			}
		}

		particles[i].weight = particle_weight;
		total_weight += particle_weight;
	}

	// Normalize:
	weights.clear();
	for(int i = 0; i < num_particles; ++i)
	{
		double weight = particles[i].weight;
		particles[i].weight = weight/total_weight;
		//std::cout << "Particle " << i << " weight: " << weight << " norm to " << particles[i].weight << std::endl;
		weights.push_back(particles[i].weight); // Add it for the resampling step
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(weights.begin(), weights.end());
    std::vector<Particle> new_particles;
    //std::cout << "ParticleFilter::resample: ";
    for(int n = 0; n < num_particles; ++n)
    {
    	int idx = d(gen);
    	new_particles.push_back(particles[idx]);
    	//std::cout << idx << ",";
    }
    //std::cout << std::endl;
    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
