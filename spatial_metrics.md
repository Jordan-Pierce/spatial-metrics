# **Spatial Metrics for optical (and acoustic?) data**

## [**Literature Review**](https://app.undermind.ai/report/0df0c798973a0130d2f8ef070bc47f663bd95600ed915dc0fec53c4ed8e2c5ca)

## **Prerequisites**

Many of the metrics listed below have specific requirements for the imagery data used, including:

1. Scale  
2. Depth Maps  
3. Orthorectification  
4. Instance and semantic segmentation masks  
   1. Labeled class categories (including biology)

Additionally, images should be collected from multiple perspectives if they are to be used for structure-from-motion. Images taken from a purely oblique angle while moving in a single direct (e.g., forward) will result in unusable data products. One way might be to collect imagery data with a tri-camera system: one forward facing, one facing nadir, and one facing the rear direction of movement.

### **Scale, Depth Maps, and Orthorectification**

An oblique image is one taken at a forward-facing direction (between a perpendicular and nadir perspective). The issues that this presents is *scale gradient*, that objects at the bottom of the image are "larger" (in pixels) than identical objects at the top, and *perspective distortion*, that circular nodules look like a flattened ellipses when viewed from an angle. Both of which can significantly skew the metrics, and therefore provide incorrect assumptions of the mapped regions. Before calculating any of the following metrics, one must conceptually or mathematically "un-tilt" the images and their oblique polygons. A common way to do this is through orthorectification, where perspective distortions are removed from images, creating a view from nadir where each pixel represents its true ground location. Combined together, these orthorectified images can be stitched to create a single, orthorectified mosaic, or “orthomosaic”. The process can be facilitated by Structure-from-Motion, though there are some *workarounds*:

* **Use Dimensionless Metrics:** Some metrics may be “safer” than others (e.g., Rugosity / Solidarity), as even though the shape is distorted by perspective, the “jaggedness” relative to the hull usually remains consistent. Others however are “unsafe” (e.g., Area and Raw Distances), where a distance at the bottom of the image is different from the top of the image.   
* **Use Region of Interest (ROIs)**: Ignore the top 30-50% of the oblique image and only calculate metrics in the bottom portion (the "near field") where the camera looks most downward. This reduces variance significantly.  
* **Apply Ground Sample Distance (GSD) Correction**: If you have altitude and camera angle, apply a simple geometric correction factor to the y-axis of your measurements based on the pixel's row index.  
* **Projection with Depth Maps:** Similar to GSD, if you have depth maps, you know the Z (distance) for every pixel. You can project the vertices of your instance polygons from image space (u, v) to world coordinates (X, Y) on the seafloor plane. A circular nodule seen obliquely looks like a flat ellipse in the image. If you calculate "Circularity" on the raw pixels, it will look deformed. If you project it to physical space first, you recover the true shape. However it is important to point out that if images have depth maps, it is assumed that you can simply orthorectify the images to obtain the orthomosaic.

Ideally 

### **Labeled Instance and Segmentation Masks**

Bounding boxes provide an important but crude localization of objects of interest in imagery. Approximate sizes can be calculated, though they are imprecise when compared to an instance segmentation mask, which acts as a tightly fitted polygon. Adding Semantic Segmentation Masks to the mix significantly enhances this type of analysis, as you are no longer approximating shapes with polygons; you now have a pixel-perfect definition of "Nodule Surface" versus "Sediment Surface." This allows you to perform differential analysis (comparing the object to its immediate background) and calculate volumetric metrics that are critical for extraction feasibility.

Having labeled data for biological organisms is the critical step for Environmental Impact Assessments (EIA). By simultaneously tracking the "Resource" (Nodules) and the "Life" (Biology), you can move beyond simple biodiversity counts to quantifying ecosystem function and habitat dependence. The central question these metrics answer is: **"***Is this organism dependent on the nodules, or does it just happen to be there***?**

## **Metrics**

* **Density & Abundance (The "How Much")**  
  * Pixel Coverage Fraction (PCF): The Visual Density Metric  
  * Mass / Resource Density (Abundance): The Economic Grade Metric  
  * Spatial Homogeneity (Quadrat Analysis): The Patchiness Metric  
  *   
* **Proximity & Clustering (The "How Close")**  
  * Edge-to-Edge Nearest Neighbor Distance (NND): The Physical Gap Metric  
  * Passability Index: The Corridor / Navigable Space Metric  
  * Ripley’s K Function (Multi-Scale Clustering): The Scale of Aggregation Metric  
  * Clark-Evans Aggregation Index: The Fingerprint Metric  
  *   
* **Individual Shape Morphology (The "What Kind")**  
  * Circularity (Isoperimetric Quotient): The Roundness Metric  
  * Rugosity (via Convexity / Solidity): The Smoothness Metric  
  * Oriented Bounding box (OBB) Aspect Ratio: The Alignment Metric  
  * Protrusion (Stick-up Height): The Collector Clearance Metric  
  *   
* **Verticality and Interaction (The “3rd Dimension”)**  
  * Protrusion (Stick-up Height): The Collector Clearance Metric  
  * Embedment Angle (Contact Slope): Breakout Force Metric  
  * Sediment Scour Anisotropy: The Environmental History Metric  
  * 3D Surface Rugosity (Texture Analysis): Sediment Magnet Metric  
  *   
* **Ecosystem Dynamics and Impact (The “Effect on Critters”)**  
  * Bivariate Ripley’s K (Cross-K Function): Invisible Halo Metric  
  * Biodiversity-Density Correlation: Habitat Value Metric  
  * Beta Diversity Turnover (Community Drift): Who Lives Where Metric  
  * Projected Biological Loss (Simulation): Reporting Metric  
  * 

### **Density & Abundance (The "How Much")**

Simple counts may be insufficient because a frame with 100 pebbles is different from a frame with 100 nodules. You need metrics that account for area coverage. These metrics answer the fundamental question: "*What is the quantity of material present*?" In the context of deep-sea mining, this determines the economic viability of a site (the resource grade) and the baseline biological habitat availability.

* **Pixel Coverage Fraction (PCF)**: **The Visual Density Metric**  
* Instead of just counting objects, *sum the area* of all polygons and divide by the total image area (or ROI area). This is the most basic metric, calculable immediately from a 2D segmentation mask or bounding boxes without any scale calibration. It represents the "*visual clutter*" of the scene.  
  * **How it Works:** It calculates the percentage of the image sensor's field of view that is occupied by the target object (nodules). This could be in pixels, or real-world units if scale is available (see below), and be calculated using polygons or segmentation masks.  
  * **Use Cases:**  
    * Real-Time "Heatmap": Since this requires no depth or scale processing, it can be computed live on the vehicle to generate a coarse heatmap of the field.  
    * Habitat Quantification: Biologists use this to measure "hard substrate availability." If PCF drops below 5%, the habitat changes from a "nodule field" to a "sediment plain," supporting different life forms.  
        
* **Mass / Resource Density (Abundance)**: **The Economic Grade Metric**  
* This is the industry-standard KPI. It moves from 2D pixels to physical weight. It requires Depth or DEMs (to estimate volume) and Scale (to normalize to square meters). The number of distinct instance centroids per unit area (e.g., if pixel-to-meter scale is known). If scale is unknown (pixels only), keep this relative.  
  * **How it Works**: It estimates the mass of every individual nodule and sums them up per unit area.  
  * **Use Cases:**   
    * Resource Reporting: This number goes directly into the feasibility study for investors. A site with a lower value might be deemed uneconomical.  
    * Collector Efficiency: By measuring this before and after the vehicle passes, you calculate the "Recovery Rate" (e.g., "We extracted 95% of the available mass").

    

* **Spatial Homogeneity (Quadrat Analysis): The Patchiness Metric**  
* Density isn't always uniform. A field might have an average of some value, but that could mean one huge pile of rocks and empty mud everywhere else. This metric quantifies that variance. Divide the image into a standardized grid (e.g., 4x4). Calculate the variance of the mineral count across these grid cells. Low variance indicates that minerals are evenly spread (homogeneous), whereas high variance indicates that minerals are clustered (patchy distribution).  
  * **How it Works:** Divide the image (or a larger map section) into a grid of equal-sized cells (quadrats). Then count the number of nodules (or mass) in each cell and compare the variance to the mean.  
  * **Use Cases:**  
    * Process Control (Engineering): A highly clustered field causes surging in the hydraulic lift system. The pump goes from sucking water (low density) to choking on rocks (high density). Knowing the homogeneity allows the control system to anticipate load changes.  
    * Geological Formation (Science): High clustering might indicate some other environmental process “moved” the nodules (e.g., underwater landslides or sorted by currents) rather than forming in place. Why else would they not form in a uniform manner?

### 

### **Proximity & Clustering (The "How Close")**

These metrics move beyond "how many" (density) to describe "how they are arranged." In deep-sea mining, this may be critical for determining machine interactions (collisions, jamming, path planning) and understanding the geological formation of the field.

* **Edge-to-Edge Nearest Neighbor Distance (NND): The Physical Gap Metric**  
* Standard spatial analysis uses centroids (center points). However, in a nodule field, a 10cm nodule and a 20cm nodule might have centroids 25cm apart, but their edges are touching. For engineering purposes (i.e., the vehicle), the gap is what matters for collection, not the center. For every mineral, calculate the distance to its single closest neighbor.  
  * **How it Works**: For every polygon, it calculates the shortest Euclidean distance from its boundary to the boundary of its closest neighbor. When using the centroid, two dots might be spaced 15cm apart, but when looking at the edges, it’s actually only 3cm.  
  * **Use Cases:**  
* Jamming Prediction (Engineering)**:** If the Edge-to-Edge distance is consistently smaller than the "intake mesh" of the collector head (e.g., \< 5cm), the nodules will bridge together and clog the intake, rather than flowing in individually.  
* Competition Analysis (Biology)**:** Sessile organisms (sponges) compete for flow. If edge-to-edge distance is zero (touching), they are competing for the same water column current.


* **Passability Index: The Corridor / Navigable Space Metric**  
* This metric inverts the analysis: instead of looking at the rocks, it looks at the empty space between them (using the semantic masks). It answers whether a vehicle or a tool can move through the field without crushing the resource or the equipment. Think “pathfinding”.  
  * **How it Works:** It analyzes the "voids" in the field. Typically calculated using a Delaunay Triangulation or a Distance Transform on the binary mask. It determines the width of the largest circle that can slide through the field without touching an object. If the outcrops are scattered randomly, passability might be high, whereas if the outcrops form a linear ridge, passability might be low.  
  * **Use Cases:**  
    * Path Planning: Automated route generation for the harvester. "Stay in the zones where Passability \> 90%."  
    * Habitat Fragmentation: If the mining leaves "stripes" of unmined area, but the Passability Index between those stripes is low for a specific animal (e.g., a large sea cucumber), the populations become genetically isolated.  
        
* **Ripley’s K Function (Multi-Scale Clustering): The Scale of Aggregation Metric**  
* Simple density indicates that things are clustered. Ripley’s K tells you how big the clusters are. It distinguishes between "small tight groups" and "large loose patches."  
  * **How it Works:** It draws a circle of radius *r* around every point and counts the neighbors. It repeats this for larger and larger values of *r* (e.g., 10cm, 50cm, 1m, 5m). Basically: “How many neighbors do I have on average with a distance of *r?”* A positive peak indicates clustering at that specific scale; negative indicates random distribution, and negative indicates dispersion or regular spacing. A peak at *r=2m* indicates nodules arranged in patches that are roughly 2m wide.  
  * **Use Cases:**  
    * Geological Origins: If clustering happens at the 10-meter scale, it might correlate with the wavelength of ancient sediment waves or seafloor topography.  
    * Operational Patch Definition: Defines the "Unit of Harvest." If patches are 2m wide, a 5m wide collector head is inefficient (it will be harvesting 60% mud). If patches are 100m wide, a wide head is efficient.  
        
* **Clark-Evans Aggregation Index: The Fingerprint Metric**  
* This is a single number that summarizes the clustering of the whole image. It is faster to calculate than Ripley’s K but less detailed.  
  * **How it Works:** It compares the Average Nearest Neighbor Distance (observed) to the Expected Distance if the points were scattered completely randomly (Poisson distribution). A value of 1 would indicate a random distribution, \> than 1 a uniform / dispersed distribution (e.g., the neighbors are further apart than random, like trees in an orchard), and \< 1 would indicate that there is a cluster (e.g., the neighbors are closer than they should be).  
  * **Use Cases:**  
    * Rapid Assessment: A quick way to tag datasets. "Site 4 is a Type R0.5 field."  
    * Biological Signaling: In biology, \> 1 (Uniformity) often implies territoriality or competition (organisms push each other away); \< 1 (Clustering) implies social behavior or shared resources (huddling for food).  
      

### **Individual Shape Morphology (The "What Kind")**

These metrics move from the "group" (density/clustering) to the "individual" (the object itself). In deep-sea mining, the specific shape of a mineral may indicate many things: how it is ingested by the vehicle, how much mud it traps, and maybe its grade. Since you have polygons, you can characterize the *type* of mineral (e.g., distinguishing a round manganese nodule from a jagged rock or crust).

* **Circularity (Isoperimetric Quotient): The Roundness Metric**  
* This is the standard geometric test for "how much like a circle is this?" However, because we are using rectified polygons (un-tilted using the depth map), we measure the physical shape, not the camera-distorted shape. Nodules tend to be formed in concentric layers and are often highly circular (values near 1.0). Fractured rocks or crusts will have lower values.  
  * **How it Works:** It compares the area of the object to its perimeter. A circle encloses the maximum area for a given perimeter. Any deviation (elongation, jagged edges) lowers the score. A score of 1 for example indicates a perfect circle, 0.7 \- 0.9 an ellipsoid (e.g., potato-shaped, nodule), and \< 0.5 a starfish, or jagged rock.  
  * **Use Cases:**  
    * Classification (Science): Differentiates **Type 1 Nodules** (discoidal / round, formed in calm water) from **Type 2 Nodules** (polynucleated / fused, formed in variable conditions).  
    * Processing Safety (Engineering): Highly non-circular objects (crust slabs) are liable to jam crushers or sorting screens designed for round stones. You can flag "Risk Zones" where circularity drops.

* **Rugosity (via Convexity / Solidity): The Smoothness Metric**  
* This is a metric for detecting defects or *jaggedness*. It ignores the overall shape (long vs. round) and focuses purely on the edges. A smooth potato-shaped nodule has high solidity. A piece of coral or a jagged rock has low solidity because the convex hull (the "rubber band" around it) will have lots of empty space. This version of rugosity does not need depth or elevation, as it is purely focused on the perimeter of the polygon.  
  * **How it Works**: It wraps a virtual "rubber band" around the object (the Convex Hull). It calculates the ratio of the object's actual area to the area inside the rubber band. A value of 1 indicates a perfectly smooth and convex surface (e.g., an egg or circle), whereas a “low” score indicates an object with deep cavities, bays, or jagged protrusions (e.g., a star or a piece of coral). Low scores might also indicate a botryoidal nodule (e.g., grapes on a vine), that when “wrapped” leaves a lot of interstitial spaces.  
  * **Use Cases:**  
    * Fragmentation Analysis: If a mining vehicle drives over a field and breaks nodules, the Solidity of the remaining pieces drops sharply (broken shards are jagged). This measures "breakage rates."  
    * Artifact Filtering: Biological artifacts (like a squid) often have tentacles or irregular limbs, resulting in very low solidity compared to rocks.

* **Oriented Bounding box (OBB) Aspect Ratio: The Alignment Metric**  
* A standard bounding box (Axis-Aligned) is useless for shape analysis because it changes if the camera rotates. The Oriented Bounding Box (OBB) finds the tightest fitting box rotated to match the object's main axis.  
  * **How it Works:** It calculates the principal axis of the polygon (using PCA or Moments) and fits a box aligned to that axis. It then compares the width (short side) to the length (long side). A value of 0 would indicate a square or circle (e.g., width is the same as length), whereas a higher value approaching 1 would indicate a long, skinny object  
  * **Use Cases:**  
    * Paleo-Current Reconstruction: If you plot the angle of the OBB for thousands of minerals, and they all align North-South, you have discovered the direction of the ancient currents that formed the field (Imbrication).  
    * Filter Feeder Detection: Stalked sponges look like long thin lines in top-down view (if they are bent over). High elongation helps segregate them from round rocks.

### **Verticality and Interaction (The “3rd Dimension”)**

Without these Z-axis metrics, you are just counting shapes. With them, you are measuring the physical reality of the seafloor—how much material sticks up, how deep it is buried, and how the ocean environment interacts with it.

* **Protrusion (Stick-up Height): The Collector Clearance Metric**  
* This is arguably the most critical operational metric for the mining vehicle. It measures the vertical exposure of the mineral above the seafloor baseline.  
* **How it Works:** It requires a Semantic Mask (to separate nodule from sediment) and a Depth Map. You mathematically "shave off" the nodules to interpolate a "Virtual Seafloor Plane," (e.g., RANSAC) then measure the height of the nodule pixels above that plane, providing a distribution of heights for every individual nodule.  
* **Use Cases:**  
  * Collector Head Settings (Engineering): If the average stick-up is 2cm, the collector head must be lowered aggressively to capture the resource, risking sediment intake. If stick-up is 10cm, the head can be raised to "clip" the nodules off the top, leaving the sediment undisturbed.  
  * Fauna Habitat (Biology): Filter feeders (like anemones) may prefer high stick-up nodules to access faster water currents. A drop in stick-up height may correlate with a drop in specific biodiversity.

* **Exposed Volume (True Mass Bridge): The Yield Metric**  
* This is the calculation that physically refines "Density." Instead of assuming a nodule is a sphere (which overestimates volume) or a flat disk (which underestimates it), this measures the actual volume protruding into the water column.  
* **How it Works:** It integrates the *Stick-up Heigh*t across the entire surface area of the nodule mask. Additionally, to get total volume, one can apply a “burial constant”, or train a machine learning model to predict the buried bottom half based on the top half’s curvature. Two nodules might have the same 2D footprint, but have different shapes, heights, etc.; based on these additional metrics, even though they look similar from a camera perspective, one might be worth extracting more than the other.  
  * **Use Cases:**  
    * Economic Valuation: This is the only way to get a true "Tonnes per square kilometer" estimate without physically collecting samples.  
    * Logistics Planning: Helps estimate the "fill rate" of the transport riser. Knowing the volume prevents clogging the vertical transport system.

* **Embedment Angle (Contact Slope): Breakout Force Metric**  
* This metric analyzes the boundary where the nodule meets the sediment. It tells you how "tightly" the seafloor is holding onto the mineral.  
  * **How it Works:** It calculates the gradient (slope) of the depth specifically at the pixels along the edge of the mask. A steep angle indicates the nodule is sitting on top of the sediment (e.g., a marble on a table), whereas a shallow angle might indicate the nodule is mounded or draped within the sediment (e.g., a buried stone). If most nodules at a site have steep angles compared to another site with shallower angles, it might be better to extract from just the first site, reducing power used by the vehicle and reducing environmental disturbance due to lower breakout force required to extract the nodules.  
  * **Use Cases:**  
    * Energy Consumption Prediction: Used to model the battery / power requirements for the vehicle.  
    * Plume Prediction: Low embedment angles imply the collector must disturb more sediment structure to extract the rock, generating a larger turbidity plume.

* **Sediment Scour Anisotropy: The Environmental History Metric**  
* This metric ignores the nodule and looks at the mud immediately surrounding it. It turns every nodule into a "current meter."  
  * **How it Works:** It measures the depth of the sediment in a ring around the nodule. It looks for asymmetry: a deep "moat" on one side (scour) and a "tail" or pile-up on the other (wake). For example, the direction vector pointing from the deep side to the shallow side may indicate current direction (think deposition on one side of a jetty, and erosion on the other).  
  * **Use Cases:**  
    * Operational Safety: Maybe you want the mining vehicle to drive into the current (to keep the plume behind it). This metric tells you the current direction from static images.  
    * Geological Validation: If the scour direction matches the orientation of the nodule's long axis (OBB), the field is in equilibrium. If they don't match, the current regime has recently changed.

* **3D Surface Rugosity (Texture Analysis): Sediment Magnet Metric**  
* Standard cameras see a nodule as a flat circle. Depth reveals that some nodules are smooth like river stones, while others are botryoidal (shaped like a bunch of grapes) or cauliflower-textured. This metric quantifies that texture intensity. It tells you how much excess surface area exists compared to the object's footprint.  
  * **How it Works:** It integrates the gradient of the depth across the nodule's surface to calculate the actual 3D surface area, and divides this by the projected 2D footprint (the pixel count in the mask). A value of 1 would indicate a perfectly smooth surface (unnaturally occurring), whereas a high value of 1.5 would indicate a highly complex surface. Additionally, if surface normals can be calculated for every pixel, if they all point “up” then the surface is smooth, else its representative of microfacets.   
  * **Use Cases:**  
    * Sediment Transport (The "Mud Proxy"): High rugosity objects trap sediment in their crevices. Harvesting Nodule B will bring up significantly more mud than Nodule A, increasing the load on the cleaning system and the size of the discharge plume.  
    * Processing Efficiency (Metallurgy): The texture often correlates with the mineral grade. In some fields, rougher "cauliflower" nodules contain higher concentrations of Nickel and Copper compared to smooth nodules (which might be iron-rich). Rugosity could serve as a non-invasive grade estimator.  
    * Micro-Habitat: In biology, high rugosity equals "high surface area" for bacteria and larvae to settle on. These are ecologically more valuable.

### **Ecosystem Dynamics and Impact**

These metrics are the core of environmental impact assessments (EIA). They move beyond counting animals to quantifying "Dependence." They answer the critical question: "If we take the rocks, do the critters die, or do they just move?" These metrics zoom out from the individual animal to look at the statistical patterns of the entire community and predict the future state of the mining site.

## 

* **Bivariate Ripley’s K (Cross-K Function): Invisible Halo Metric**  
* This a standard for proving "spatial dependence" in ecology. It statistically answers: "Are the animals clustering around the nodules more than random chance would predict?" It detects relationships that are invisible to the naked eye.  
  * **How it Works:** You take two sets of points: Set A (Nodule Centroids) and Set B (Biological Centroids). You count how many Type B points fall within distance *r* of every Type A point. You compare this to a "Random Null Model" (simulating random points). If the line is above, this indicates a positive association (attraction), that organisms prefer the nodules; If the line is within, this would indicate a random association (no relationship), and finally, if the line is below, this indicates a segregation (repulsion) indicating that the organisms avoid nodules. Even animals that are not touching the nodules (e.g., swimming fish) prefer to stay within a short distance, it is likely an attraction (e.g., food). Thus the ecological footprint is wider than each nodule itself.  
  * **Use Cases:**  
    * Defining Buffer Zones: If the "Halo" extends 2 meters, then your mining buffers need to be larger than 2 meters to protect the edge effects.  
    * Restoration Metrics: When monitoring a post-mining site, if the Ripley's K curve remains flat (random), the ecosystem structure has not returned.

* **Biodiversity-Density Correlation: Habitat Value Metric**  
* This metric tests the "More Rocks \== More Life" hypothesis. It quantifies the richness of the ecosystem as a function of the resource grade.  
  * **How it Works**: You divide the survey track into grid cells; in each, you calculate Nodule Density (X-axis) and Biological Abundance/Richness (Y-axis). You then run a Pearson or Spearman correlation. A high value of 1 indicates a strong positive link: if the desired mining area is also a high-biodiversity region, this creates conflict. Whereas if the value is low, this indicates that biology might exist for some other reason, and extracting nodules may not have an impact on the community.  
  * **Use Cases:**  
    * Strategic Planning: Companies prefer sites with low correlation (high resource, low bio-overlap).  
    * Baselines: Establishes the pre-mining baseline. If the correlation breaks after a test mining run, you have fundamentally altered the ecosystem driver.

* **Beta Diversity Turnover (Community Drift): Who Lives Where Metric**  
* Simple counts ("Alpha Diversity") are misleading. You might have 100 animals in Zone A and 100 in Zone B, but are they the same animals? Beta Diversity measures how the identity of species changes as you move across the nodule field.  
  * **How it Works:** It compares the species list of one track segment to the next. A value of 0 indicates that the communities are identical, whereas a value of 1 indicates completely different species.  
  * **Use Cases:**  
    * Zoning: If Beta Diversity is high, you cannot just preserve one corner of the block and assume you saved everything. You need a network of preservation zones to capture the changing community.

* **Projected Biological Loss (Simulation): Reporting Metric**  
* This is not a measurement of the present, but a calculation of the future.  
  * **How it Works:** It combines the Substrate Occupancy rules with the proposed Mining Plan.   
    * Load the Mining Path (Polygon of area to be removed).  
    * Identify all "Obligate" organisms inside that path (100% mortality).  
    * Identify all "Shelter-Seeking" organisms within *X* meters of the path (Predicted partial mortality).  
    * Sum the biomass.  
  * **Scenario:**  
    * A Mining plan cuts through the dense center of a field  
    * Removal of 5K tonnes of nodules will result in the direct loss of 450KG of biomass, impacting *X* species.  
  * **Use Cases:**  
    * Scenario Comparison: Compare "Plan A" (strip mining) vs "Plan B" (checkerboard mining). Which one kills less biomass for the same amount of ore?  
    * Stakeholder Communication: Converts abstract ecological concepts into concrete "Loss vs. Gain" numbers.

---

# 

# Key Takeaways for Insight Generation

(ideas, need work)

### **1\. The "Breakout Force" Proxy (Energy Efficiency)**

*Combinations: Embedment Angle \+ Nodule Size*

* **The Logic:** Combines the weight of the object with how "buried" it is in the sediment.  
* **Insight:**  
  * **High Cost:** Large, deeply embedded nodules (shallow contact angle). These require high suction or mechanical force to dislodge.  
  * **Low Cost:** Small, high stick-up nodules. These sit loosely on the surface and are easily collected.

### **2\. The "Sediment Load" Proxy (Processing Efficiency)**

*Combinations: 3D Rugosity \+ Protrusion*

* **The Logic:** Combines the "stickiness" of the nodule's texture with its depth in the mud.  
* **Insight:**  
  * **Worst Case:** Low protrusion \+ High rugosity. The nodule is buried deep in mud, and its complex texture traps sediment like a sponge. This maximizes the load on the riser system and cleaning equipment.  
  * **Best Case:** Smooth nodules sitting high on the sediment surface.

### **3\. The "Safety" Proxy (Equipment Risk)**

*Combinations: Passability Index \+ True Circularity*

* **The Logic:** Combines the density of obstacles with the "sharpness" of those obstacles.  
* **Insight:**  
  * **High Risk:** Low passability (crowded) \+ Low circularity (jagged/sharp). This indicates a field of tightly packed, sharp rocks or crusts that can damage collector heads or hoses.  
  * **Low Risk:** High passability \+ High circularity (round nodules with space between them).

### **4\. The "Conservation Value" Index (Conflict Heatmap)**

*Combinations: Mass Yield ($kg/m^2$) \+ Substrate Occupancy %*

* **The Logic:** Normalizes the economic value against the ecological cost.  
* **Insight:**  
  * **Red Zone (High Conflict):** High resource density but high biological dependence. Mining here causes maximum ecological damage per dollar earned.  
  * **Green Zone (Low Conflict):** High resource density with low biological dependence (few animals, or animals not reliant on the nodules). These are optimal "First Harvest" sites.

### **5\. The "Plume Potential" Indicator (Turbidity Risk)**

*Combinations: Embedment Angle \+ Sediment Scour Anisotropy*

* **The Logic:** Combines how deep the collector must dig with how loose/mobile the surrounding sediment is.  
* **Insight:**  
  * **High Plume Risk:** Nodules are buried deep (low angle) in loose, scour-prone sediment. This signals a need for active plume mitigation (e.g., slowing down) to prevent massive turbidity clouds.

### **6\. The "Harvest Rhythm" Predictor (Mechanical Load)**

*Combinations: Spatial Homogeneity \+ Size Class Variance*

* **The Logic:** Predicts the consistency of the material entering the collector.  
* **Insight:**  
  * **Surging Load:** A "patchy" field (High Variance) with mixed sizes creates a fluctuating load—the intake goes from empty to jammed in seconds. This informs the hydraulic control logic (e.g., "Prepare for pressure spikes").

### **7\. The "Habitat Connectivity" Warning (Ecosystem Fragmentation)**

*Combinations: Passability (Corridor Width) \+ Bivariate Proximity (Shelter Range)*

* **The Logic:** Compares the width of the mining track to the distance animals are willing to travel from safety.  
* **Insight:**  
  * **Island Effect:** If the harvested path is wider than the distance small animals will travel across open sediment, you create isolated "biological islands," halting genetic flow.

### **8\. The "Paleo-Current" Reconstruction (Geological Validation)**

*Combinations: OBB Orientation \+ Sediment Scour Direction*

* **The Logic:** Compares the ancient alignment of the nodules with the current erosion patterns.  
* **Insight:**  
  * **Aligned:** The current regime has been stable for millennia.  
  * **Misaligned:** The current that *formed* the field is different from the current *scouring* it today. This helps geologists understand the age and stability of the deposit.

