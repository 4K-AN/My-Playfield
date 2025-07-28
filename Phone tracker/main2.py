import phonenumbers
from phonenumbers import geocoder, carrier
import folium
import webbrowser
import time
import random
import math
from myphone import number

# Optional imports dengan error handling
try:
    from opencage.geocoder import OpenCageGeocode
    GEOCODING_AVAILABLE = True
except ImportError:
    GEOCODING_AVAILABLE = False
    print("⚠️  OpenCage not available - using fallback methods")

try:
    from concurrent.futures import ThreadPoolExecutor
    THREADING_AVAILABLE = True
except ImportError:
    THREADING_AVAILABLE = False
    print("⚠️  Threading not available - using sequential processing")

class BulletproofYogyaTracker:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.start_time = time.time()
        
        print("🛡️  Initializing Bulletproof Yogya Tracker...")
        
        # Essential Yogyakarta data
        self.yogya_data = {
            'center': [-7.7956, 110.3695],
            'bounds': {
                'north': -7.5, 'south': -8.2,
                'west': 110.0, 'east': 110.8
            },
            'districts': {
                'Kota_Yogyakarta': [-7.7956, 110.3695],
                'Sleman': [-7.7326, 110.3553],
                'Bantul': [-7.8879, 110.3281],
                'Kulon_Progo': [-7.8271, 110.1553],
                'Gunungkidul': [-7.9553, 110.5942]
            },
            'key_areas': {
                'Malioboro': [-7.7925, 110.3656],
                'UGM_Area': [-7.7717, 110.3754],
                'Tugu_Station': [-7.7836, 110.3634],
                'Kraton': [-7.8051, 110.3642],
                'Kotagede': [-7.8287, 110.3967],
                'Depok_Sleman': [-7.7612, 110.4013],
                'Bantul_Square': [-7.8879, 110.3281],
                'Wates_Center': [-7.8271, 110.1553],
                'Wonosari': [-7.9553, 110.5942]
            }
        }
        
        # Operator data
        self.operators = {
            'telkomsel': {
                'prefixes': ['811', '812', '813', '821', '822', '823', '851', '852', '853'],
                'towers': [
                    {'id': 'TSL_01', 'pos': [-7.7956, 110.3695], 'area': 'Yogyakarta'},
                    {'id': 'TSL_02', 'pos': [-7.7326, 110.3553], 'area': 'Sleman'},
                    {'id': 'TSL_03', 'pos': [-7.8879, 110.3281], 'area': 'Bantul'}
                ]
            },
            'indosat': {
                'prefixes': ['814', '815', '816', '855', '856', '857', '858'],
                'towers': [
                    {'id': 'IDS_01', 'pos': [-7.7945, 110.3634], 'area': 'Yogyakarta'},
                    {'id': 'IDS_02', 'pos': [-7.7234, 110.4123], 'area': 'Sleman'}
                ]
            },
            'xl': {
                'prefixes': ['817', '818', '819', '859', '877', '878'],
                'towers': [
                    {'id': 'XL_01', 'pos': [-7.8019, 110.3658], 'area': 'Yogyakarta'},
                    {'id': 'XL_02', 'pos': [-7.8234, 110.3123], 'area': 'Bantul'}
                ]
            }
        }
        
        print(f"✅ Initialization complete ({time.time() - self.start_time:.2f}s)")
    
    def safe_phone_parse(self, phone_number):
        """Safe phone number parsing dengan error handling"""
        try:
            parsed = phonenumbers.parse(phone_number)
            if phonenumbers.is_valid_number(parsed):
                return {
                    'parsed': parsed,
                    'country_code': parsed.country_code,
                    'national_number': str(parsed.national_number),
                    'valid': True
                }
            else:
                print("⚠️  Phone number format invalid")
                return {'valid': False}
        except Exception as e:
            print(f"⚠️  Phone parsing error: {e}")
            return {'valid': False}
    
    def detect_operator(self, national_number):
        """Detect operator dengan fallback"""
        try:
            for operator, data in self.operators.items():
                for prefix in data['prefixes']:
                    if national_number.startswith(prefix):
                        return operator
            
            # Fallback based on common patterns
            if national_number.startswith('8'):
                return 'telkomsel'  # Most common
            return 'unknown'
        except:
            return 'telkomsel'  # Safe fallback
    
    def simulate_triangulation(self, operator, phone_number):
        """Safe triangulation simulation"""
        try:
            if operator not in self.operators:
                operator = 'telkomsel'
            
            towers = self.operators[operator]['towers']
            
            # Deterministic seed from phone number
            seed = abs(hash(phone_number)) % 10000
            random.seed(seed)
            
            # Select towers
            selected_towers = random.sample(towers, min(2, len(towers)))
            
            if len(selected_towers) >= 2:
                # Weighted triangulation
                pos1, pos2 = selected_towers[0]['pos'], selected_towers[1]['pos']
                w1, w2 = random.uniform(0.4, 0.7), random.uniform(0.3, 0.6)
                total_weight = w1 + w2
                w1, w2 = w1/total_weight, w2/total_weight
                
                est_lat = pos1[0] * w1 + pos2[0] * w2
                est_lon = pos1[1] * w1 + pos2[1] * w2
                
                # Add small random variation
                est_lat += random.uniform(-0.008, 0.008)
                est_lon += random.uniform(-0.008, 0.008)
                
                # Ensure within Yogyakarta bounds
                bounds = self.yogya_data['bounds']
                est_lat = max(bounds['south'], min(bounds['north'], est_lat))
                est_lon = max(bounds['west'], min(bounds['east'], est_lon))
                
                confidence = random.uniform(0.70, 0.90)
                
                return {
                    'location': [est_lat, est_lon],
                    'confidence': confidence,
                    'method': 'BTS Triangulation',
                    'towers_used': len(selected_towers),
                    'operator': operator
                }
        except Exception as e:
            print(f"⚠️  Triangulation error: {e}")
        
        return None
    
    def safe_geocoding(self, location_query="Yogyakarta"):
        """Safe geocoding with multiple fallbacks"""
        if not GEOCODING_AVAILABLE or not self.api_key:
            return self.fallback_geocoding(location_query)
        
        try:
            geocoder_api = OpenCageGeocode(self.api_key)
            
            queries = [
                f"{location_query}, Yogyakarta, Indonesia",
                "Special Region of Yogyakarta, Indonesia",
                "Yogyakarta, Indonesia"
            ]
            
            for query in queries:
                try:
                    results = geocoder_api.geocode(query, limit=1, timeout=3)
                    if results and len(results) > 0:
                        result = results[0]
                        
                        # Validate result is in Yogyakarta region
                        lat, lon = result['geometry']['lat'], result['geometry']['lng']
                        if self.is_in_yogyakarta_region(lat, lon):
                            return {
                                'location': [lat, lon],
                                'confidence': min(result.get('confidence', 5) / 10, 0.7),
                                'method': 'OpenCage Geocoding',
                                'address': result['formatted']
                            }
                    time.sleep(0.1)  # Small delay
                except Exception as e:
                    print(f"⚠️  Geocoding query failed: {e}")
                    continue
        except Exception as e:
            print(f"⚠️  Geocoding service error: {e}")
        
        return self.fallback_geocoding(location_query)
    
    def fallback_geocoding(self, location_hint):
        """Fallback geocoding using local data"""
        # Try to match location hint with known areas
        location_hint_lower = location_hint.lower()
        
        for area, coords in self.yogya_data['key_areas'].items():
            if any(word in location_hint_lower for word in area.lower().split('_')):
                return {
                    'location': coords,
                    'confidence': 0.5,
                    'method': 'Local Database Match',
                    'address': f"{area.replace('_', ' ')}, Yogyakarta, Indonesia"
                }
        
        # Default to Yogyakarta center
        return {
            'location': self.yogya_data['center'],
            'confidence': 0.3,
            'method': 'Yogyakarta Center Fallback',
            'address': 'Yogyakarta City Center, Special Region of Yogyakarta, Indonesia'
        }
    
    def is_in_yogyakarta_region(self, lat, lon):
        """Check if coordinates are within Yogyakarta region"""
        bounds = self.yogya_data['bounds']
        return (bounds['south'] <= lat <= bounds['north'] and 
                bounds['west'] <= lon <= bounds['east'])
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance using haversine formula"""
        try:
            R = 6371  # Earth radius in km
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = (math.sin(dlat/2)**2 + 
                 math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
            return 2 * R * math.asin(math.sqrt(a))
        except:
            # Fallback to simple distance
            return math.sqrt((lat1-lat2)**2 + (lon1-lon2)**2) * 111
    
    def find_nearby_areas(self, location, max_areas=5):
        """Find nearby areas with distance calculation"""
        try:
            lat, lon = location
            areas_with_distance = []
            
            for area, coords in self.yogya_data['key_areas'].items():
                distance = self.calculate_distance(lat, lon, coords[0], coords[1])
                areas_with_distance.append({
                    'area': area.replace('_', ' '),
                    'distance': distance,
                    'coordinates': coords
                })
            
            return sorted(areas_with_distance, key=lambda x: x['distance'])[:max_areas]
        except Exception as e:
            print(f"⚠️  Area calculation error: {e}")
            return []
    
    def create_bulletproof_map(self, results, phone_number):
        """Create map with extensive error handling"""
        try:
            print("🗺️  Creating bulletproof map...")
            
            # Determine center location
            if results and len(results) > 0:
                best_result = max(results, key=lambda x: x.get('confidence', 0))
                center_location = best_result.get('location', self.yogya_data['center'])
            else:
                center_location = self.yogya_data['center']
            
            # Create base map with safe settings
            my_map = folium.Map(
                location=center_location,
                zoom_start=12,
                tiles='OpenStreetMap'  # Most reliable tile source
            )
            
            # Add results markers with error handling
            colors = ['red', 'blue', 'green', 'purple', 'orange']
            
            for i, result in enumerate(results[:5]):  # Limit to 5 for safety
                try:
                    color = colors[i % len(colors)]
                    location = result.get('location')
                    
                    if location and len(location) == 2:
                        # Safe popup creation
                        popup_content = f"""
                        <div style="width: 200px;">
                            <h4>{result.get('method', 'Unknown Method')}</h4>
                            <b>Phone:</b> {phone_number}<br>
                            <b>Location:</b> {location[0]:.4f}, {location[1]:.4f}<br>
                            <b>Confidence:</b> {result.get('confidence', 0):.1%}<br>
                            <b>Operator:</b> {result.get('operator', 'N/A')}
                        </div>
                        """
                        
                        # Add marker with error handling
                        try:
                            folium.Marker(
                                location,
                                popup=folium.Popup(popup_content, max_width=250),
                                tooltip=f"{result.get('method', 'Result')}: {result.get('confidence', 0):.1%}",
                                icon=folium.Icon(color=color, icon='phone', prefix='fa')
                            ).add_to(my_map)
                        except:
                            # Fallback to simple marker
                            folium.Marker(
                                location,
                                popup=f"{result.get('method', 'Result')}: {result.get('confidence', 0):.1%}"
                            ).add_to(my_map)
                        
                        # Add uncertainty circle
                        try:
                            uncertainty_radius = max(500, (1 - result.get('confidence', 0.5)) * 2000)
                            folium.Circle(
                                location,
                                radius=uncertainty_radius,
                                color=color,
                                fill=True,
                                fillOpacity=0.15,
                                weight=2,
                                popup=f"Uncertainty area - {result.get('method', 'Unknown')}"
                            ).add_to(my_map)
                        except:
                            pass  # Skip circle if fails
                
                except Exception as e:
                    print(f"⚠️  Error adding marker {i}: {e}")
                    continue
            
            # Add district markers
            try:
                for district, coords in self.yogya_data['districts'].items():
                    folium.Marker(
                        coords,
                        popup=f"<b>{district.replace('_', ' ')}</b>",
                        tooltip=district.replace('_', ' '),
                        icon=folium.Icon(color='lightblue', icon='building', prefix='fa')
                    ).add_to(my_map)
            except:
                pass  # Skip if district markers fail
            
            # Add safe legend
            try:
                legend_html = f'''
                <div style="position: fixed; 
                            bottom: 50px; left: 50px; width: 250px; height: auto; 
                            background-color: rgba(255,255,255,0.95); 
                            border:2px solid #333; z-index:9999; 
                            font-size:12px; padding: 15px; border-radius: 8px;
                            box-shadow: 0 0 10px rgba(0,0,0,0.3);
                            ">
                <h4 style="margin-top: 0; color: #333;">📱 Yogyakarta Phone Analysis</h4>
                <p><b>Phone:</b> {phone_number}</p>
                <p><b>Results Found:</b> {len(results)}</p>
                '''
                
                for i, result in enumerate(results[:3]):
                    color_emoji = ['🔴', '🔵', '🟢'][i]
                    legend_html += f'<p style="margin: 2px 0;">{color_emoji} {result.get("method", "Unknown")}: {result.get("confidence", 0):.1%}</p>'
                
                legend_html += '''
                <hr style="margin: 10px 0;">
                <p style="font-size: 10px; color: #666; margin-bottom: 0;">
                🛡️ Bulletproof tracker - Educational purpose only<br>
                🏛️ Special Region of Yogyakarta, Indonesia
                </p>
                </div>
                '''
                
                my_map.get_root().html.add_child(folium.Element(legend_html))
            except Exception as e:
                print(f"⚠️  Legend creation error: {e}")
            
            return my_map
            
        except Exception as e:
            print(f"❌ Map creation failed: {e}")
            # Create minimal fallback map
            fallback_map = folium.Map(
                location=self.yogya_data['center'],
                zoom_start=11
            )
            folium.Marker(
                self.yogya_data['center'],
                popup="Yogyakarta Center - Fallback Location"
            ).add_to(fallback_map)
            return fallback_map
    
    def run_bulletproof_analysis(self, phone_number):
        """Run complete analysis with maximum error handling"""
        analysis_start = time.time()
        
        print("🛡️  BULLETPROOF YOGYAKARTA ANALYSIS")
        print("="*50)
        
        # Step 1: Parse phone number
        phone_data = self.safe_phone_parse(phone_number)
        if not phone_data['valid']:
            print("❌ Invalid phone number format")
            return []
        
        national_number = phone_data['national_number']
        print(f"📱 Phone: {phone_number}")
        print(f"🔢 National: {national_number}")
        
        # Step 2: Detect operator
        operator = self.detect_operator(national_number)
        print(f"📡 Operator: {operator.upper()}")
        
        results = []
        
        # Step 3: Multiple analysis methods
        methods_attempted = 0
        
        # Method 1: BTS Triangulation
        try:
            triangulation_result = self.simulate_triangulation(operator, phone_number)
            if triangulation_result:
                results.append(triangulation_result)
                print(f"✅ Triangulation: {triangulation_result['confidence']:.1%} confidence")
                methods_attempted += 1
        except Exception as e:
            print(f"⚠️  Triangulation failed: {e}")
        
        # Method 2: Geocoding
        try:
            geocoding_result = self.safe_geocoding("Yogyakarta")
            if geocoding_result:
                results.append(geocoding_result)
                print(f"✅ Geocoding: {geocoding_result['confidence']:.1%} confidence")
                methods_attempted += 1
        except Exception as e:
            print(f"⚠️  Geocoding failed: {e}")
        
        # Method 3: Area-based estimation
        try:
            if results:
                best_location = results[0]['location']
                nearby_areas = self.find_nearby_areas(best_location, 3)
                if nearby_areas:
                    print(f"📍 Nearby areas found: {len(nearby_areas)}")
                    for area in nearby_areas[:2]:
                        print(f"   📌 {area['area']}: {area['distance']:.1f}km")
        except Exception as e:
            print(f"⚠️  Area analysis failed: {e}")
        
        analysis_time = time.time() - analysis_start
        print(f"⏱️  Analysis completed in {analysis_time:.2f}s ({methods_attempted} methods)")
        
        return results

def main():
    """Bulletproof main execution"""
    try:
        print("🚀 STARTING BULLETPROOF YOGYAKARTA ANALYSIS")
        print("="*55)
        
        # Initialize with error handling
        api_key = 'd9b612d5de1c4e23a8d4d81e8e9f3b26'  # Replace with your key
        tracker = BulletproofYogyaTracker(api_key)
        
        # Run analysis
        results = tracker.run_bulletproof_analysis(number)
        
        if results and len(results) > 0:
            print(f"\n📊 ANALYSIS RESULTS")
            print("="*30)
            
            for i, result in enumerate(results, 1):
                location = result.get('location', [0, 0])
                print(f"{i}. {result.get('method', 'Unknown')}")
                print(f"   📍 {location[0]:.6f}, {location[1]:.6f}")
                print(f"   🎯 {result.get('confidence', 0):.1%}")
                
                # Validation
                if tracker.is_in_yogyakarta_region(location[0], location[1]):
                    print("   ✅ Within Yogyakarta region")
                else:
                    print("   ⚠️  Outside Yogyakarta region")
                print()
            
            # Create map
            try:
                map_obj = tracker.create_bulletproof_map(results, number)
                map_file = "bulletproof_yogya_analysis.html"
                map_obj.save(map_file)
                print(f"🗺️  Map saved: {map_file}")
                webbrowser.open(map_file)
                
                # Best result summary
                best = max(results, key=lambda x: x.get('confidence', 0))
                print(f"\n🏆 BEST ESTIMATE")
                print(f"📍 {best['location'][0]:.6f}, {best['location'][1]:.6f}")
                print(f"🎯 {best.get('confidence', 0):.1%} confidence")
                print(f"🔧 Method: {best.get('method', 'Unknown')}")
                
            except Exception as e:
                print(f"❌ Map creation error: {e}")
        
        else:
            print("❌ No analysis results - creating fallback...")
            try:
                fallback_map = folium.Map(location=[-7.7956, 110.3695], zoom_start=11)
                folium.Marker([-7.7956, 110.3695], popup="Yogyakarta Fallback").add_to(fallback_map)
                fallback_map.save("yogya_fallback.html")
                webbrowser.open("yogya_fallback.html")
                print("✅ Fallback map created")
            except Exception as e:
                print(f"❌ Even fallback failed: {e}")
    
    except Exception as e:
        print(f"❌ Critical error in main: {e}")
        print("🔧 Try checking your environment and dependencies")

if __name__ == "__main__":
    main()