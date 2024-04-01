#######################
# Import libraries
import streamlit as st
import plotly.express as px
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
import json
from IPython.display import display
import streamlit.components.v1 as components
from folium import plugins
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import h3
import libpysal as lps
from libpysal.weights import Queen 
import esda
from splot.esda import plot_moran, moran_scatterplot, lisa_cluster
from esda.moran import Moran, Moran_Local
from streamlit_folium import folium_static
import plotly.graph_objects as go
from folium.plugins import Fullscreen
import seaborn as sns
import matplotlib.colors as mcolors

#######################
# Page configuration
st.set_page_config(
    page_title="국립공원 Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")
#######################

# Plots
def find_boundary(npark_boundary,npark_name):
    npark_boundary = npark_boundary[npark_boundary['DESIG']=='국립공원']
    seoul_npark_boundary = npark_boundary[npark_boundary['ORIG_NAME']==npark_name]
    return seoul_npark_boundary

# Plots
def find_boundary_hotspot(npark_boundary,npark_name):
    npark_boundary = npark_boundary[npark_boundary['DESIG']=='국립공원']
    seoul_npark_boundary = npark_boundary[npark_boundary['ORIG_NAME']==npark_name]
    name_list = ['설악산','변산반도','경주','덕유산','다도해해상','월악산','오대산','한려해상','태안해안']
    if npark_name in name_list:
        seoul_npark_boundary=seoul_npark_boundary.explode()
        seoul_npark_boundary = seoul_npark_boundary.reset_index()
        seoul_npark_boundary = seoul_npark_boundary[seoul_npark_boundary.index==0]
    
    return seoul_npark_boundary

def sjoin(gdf_park_data,npark_boundary,npark_name):
    gdf_seoul_park_data = gdf_park_data[gdf_park_data['국립공원명']==npark_name]
    seoul_accident = gpd.sjoin(gdf_seoul_park_data, npark_boundary)
    return seoul_accident

def find_center_latitudes_longtitudes(accident):
    latitudes = accident['위도_변환'].tolist()
    longitudes = accident['경도_변환'].tolist()
    map_center_lat = sum(latitudes) / len(latitudes)
    map_center_lon = sum(longitudes) / len(longitudes)
    return map_center_lat,map_center_lon

def v_world(selected_national_park_accident):
    map_center_lat,map_center_lon = find_center_latitudes_longtitudes(selected_national_park_accident)

    tiles = f"http://api.vworld.kr/req/wmts/1.0.0/{vworld_key}/{layer}/{{z}}/{{y}}/{{x}}.{tileType}"
    attr = "Vworld"
    # 기본 지도 객체 생성
    m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=12,tiles=tiles, attr=attr)
    # VWorld Hybrid 타일 추가
    satelitelayer = folium.TileLayer(
        tiles=f'http://api.vworld.kr/req/wmts/1.0.0/{vworld_key}/Hybrid/{{z}}/{{y}}/{{x}}.png',
        attr='VWorld Hybrid',
        name='지명표시',
        overlay=True
    ).add_to(m)
    return m

def make_pointplot(selected_national_park_accident,selected_npark_boundary):    
    # 사고 원인별 색상 사전 정의
    palette = sns.color_palette('bright')

    # 사건 유형에 대한 색상 딕셔너리
    color_dict = {
        '실족': palette[0],
        '기타': palette[1],
        '일시적고립': palette[2],
        '탈진경련': palette[3],
        '부주의': palette[4],
        '추락': palette[5],
        '심장문제': palette[6],
        '해충 및 동물피해': palette[7],
        '낙하물': palette[8],
    }

    # 컬러 팔레트에 해당하는 RGB 값을 hex 코드로 변환
    color_dict_hex = {key: mcolors.rgb2hex(value) for key, value in color_dict.items()}



    map_center_lat, map_center_lon = find_center_latitudes_longtitudes(selected_national_park_accident)

    tiles = f"http://api.vworld.kr/req/wmts/1.0.0/{vworld_key}/{layer}/{{z}}/{{y}}/{{x}}.{tileType}"
    attr = "Vworld"
    # 기본 지도 객체 생성
    m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=12)
    # 전체 화면 버튼 추가
    fullscreen = Fullscreen(position='topleft',  # 버튼 위치
                            title='전체 화면',     # 마우스 오버시 표시될 텍스트
                            title_cancel='전체 화면 해제',  # 전체 화면 모드 해제 버튼의 텍스트
                            force_separate_button=True)  # 전체 화면 버튼을 별도의 버튼으로 표시
    m.add_child(fullscreen)
    # VWorld Hybrid 타일 추가
    satelitelayer = folium.TileLayer(
        tiles=tiles,
        attr='VWorld Hybrid',
        name='위성사진',
        overlay=True
    ).add_to(m)
    
    # 사고 원인별로 레이어 그룹 생성 및 추가
    for i in color_dict_hex.keys(): # color_dict를 사용하여 반복
        # 사고 원인별 데이터 필터링
        type_accident = selected_national_park_accident[selected_national_park_accident['유형'] == i]
        
        # 해당 사고 원인의 색상 가져오기
        accident_color = color_dict_hex[i]  # 사고 원인별로 정의된 색상 사용
        
        # 사고 원인별 레이어 그룹 생성
        layer_group = folium.FeatureGroup(name=i)
        
        # 사고 위치에 대한 CircleMarker 추가 및 툴팁 정보 설정
        for idx, row in type_accident.iterrows():
            tooltip_text = f"유형: {row['유형']}<br>사고 일자: {row['국립공원명']}"  # 툴팁 텍스트 정의
            popup_text = f"유형: {row['유형']}<br>사고 일자: {row['국립공원명']}<br>위치: {row['위도_변환']}, {row['경도_변환']}"
            folium.CircleMarker(
                location=(row['위도_변환'], row['경도_변환']),
                radius=3,
                color=accident_color,
                fill=True,
                fill_color=accident_color,
                fill_opacity=1.0,  # 내부 채움 불투명도
                popup=popup_text,
                tooltip=tooltip_text  # 툴팁 추가
            ).add_to(layer_group)
        
        # 레이어 그룹을 지도 객체에 추가
        layer_group.add_to(m)
    geojson_data = json.loads(selected_npark_boundary.to_json())
    folium.GeoJson(
        geojson_data,
        name='국립공원 경계',
        style_function=lambda feature: {
            'color': 'yellow',
            'weight': 2,
            'fillOpacity': 0
        }
    ).add_to(m)

    # 레이어 컨트롤 추가
    folium.LayerControl().add_to(m)
    # 범례 추가 코드 ...
    # 범례 추가
    legend_html = '''
     <div style="position: fixed; 
     bottom: 5px; left: 5px; width: 150px; height: 150px; 
     background-color: white; color: black; border:2px solid grey; z-index:9999; font-size:14px;
     padding: 1px 1px; ">
     &nbsp; 실족추락 &nbsp; <i class="fa fa-circle fa-1x" style="color:red"></i><br>
     &nbsp; 무리한 활동 &nbsp; <i class="fa fa-circle fa-1x" style="color:green"></i><br>
     &nbsp; 길잃음 &nbsp; <i class="fa fa-circle fa-1x" style="color:blue"></i><br>
     &nbsp; 개인질환 &nbsp; <i class="fa fa-circle fa-1x" style="color:yellow"></i><br>
     &nbsp; 동물 및 해충 &nbsp; <i class="fa fa-circle fa-1x" style="color:gray"></i><br>
     &nbsp; 부주의 &nbsp; <i class="fa fa-circle fa-1x" style="color:purple"></i><br>
     &nbsp; 자연 및 환경 &nbsp; <i class="fa fa-circle fa-1x" style="color:skyblue"></i><br>
      </div>
     '''
    m.get_root().html.add_child(folium.Element(legend_html))

    return m






########################################### 핫스팟과 안전쉼터가 입력한 거리보다 넘어간 핫스팟 위치 표시 ##################################
def make_hotspot_safetyplace(selected_national_park_accident,selected_npark_boundary,safety_place,distance=500):
    single_polygon = selected_npark_boundary['geometry'].unary_union
    # GeoJSON 형식으로 변환
    geojson_polygon = single_polygon.__geo_interface__
    resolution=9
    # 북한산 국립공원을 커버하는 H3 육각형 인덱스 생성
    hexagons = list(h3.polyfill_geojson(geojson_polygon, resolution))

    # polygon = seoul_npark_boundary.iloc[0]['geometry']
    # 육각형 경계 좌표 생성
    hexagon_polygons = [Polygon(h3.h3_to_geo_boundary(hex, geo_json=True)) for hex in hexagons]

    # GeoDataFrame 생성
    gdf_polygon = gpd.GeoDataFrame(geometry=hexagon_polygons)

    prices = gpd.sjoin(gdf_polygon, selected_national_park_accident[['geometry']], op='contains')
    
    # 각 geometry에 대한 카운트를 계산
    nbr_avg_price = prices.groupby('geometry').size().reset_index(name='acc_counts')
    # 병합하기
    nbr_final = gdf_polygon.merge(nbr_avg_price, on='geometry', how='left')
    # 병합 결과에서 결측값을 0으로 채우기
    nbr_final['acc_counts'] = nbr_final['acc_counts'].fillna(0)
    w =  lps.weights.Queen.from_dataframe(nbr_final)
    w.transform = 'r'

    # 속성 유사성(Attribute similarity) / 가중치 적용 사고발생건수 추가
    nbr_final['weighted_acc_counts'] = lps.weights.lag_spatial(w, nbr_final['acc_counts'])

    # 광역적 공간 자기상관 / 상관계수값 출력 
    y = nbr_final.acc_counts
    moran = esda.Moran(y, w)
    moran_local = Moran_Local(y, w)
    nbr_final['Cluster'] = moran_local.q
    cluster_map = {1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}
    nbr_final['Cluster_Label'] = nbr_final['Cluster'].map(cluster_map)
    moran_local = Moran_Local(y, w)
    # p-value 기준으로 유의미한 결과만 필터링
    p_threshold = 0.05  # 유의수준 설정
    nbr_final['Significant'] = moran_local.p_sim < p_threshold

    # 유의미하지 않은 관측치에 대해 'NS' 라벨 할당
    nbr_final['Cluster_Label'] = nbr_final.apply(lambda row: 'NS' if not row['Significant'] else row['Cluster_Label'], axis=1)

    # 'Significant' 열은 이제 필요 없으므로 제거하거나, 유지하려면 이 단계를 생략
    nbr_final.drop(columns=['Significant'], inplace=True)

    center = nbr_final.geometry.centroid.unary_union.centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=12)
    # 전체 화면 버튼 추가
    fullscreen = Fullscreen(position='topleft',  # 버튼 위치
                            title='전체 화면',     # 마우스 오버시 표시될 텍스트
                            title_cancel='전체 화면 해제',  # 전체 화면 모드 해제 버튼의 텍스트
                            force_separate_button=True)  # 전체 화면 버튼을 별도의 버튼으로 표시
    m.add_child(fullscreen)
    # VWorld Satellite Layer 추가
    vworld_key = "BF677CB9-D1EA-3831-B328-084A9AE3CDCC"
    satellite_layer = folium.TileLayer(
        tiles=f"http://api.vworld.kr/req/wmts/1.0.0/{vworld_key}/Satellite/{{z}}/{{y}}/{{x}}.jpeg",
        attr='VWorld Satellite', 
        name='위성지도'
    ).add_to(m)

    # VWorld Hybrid Layer 추가
    hybrid_layer = folium.TileLayer(
        tiles=f"http://api.vworld.kr/req/wmts/1.0.0/{vworld_key}/Hybrid/{{z}}/{{y}}/{{x}}.png",
        attr='VWorld Hybrid', 
        name='지명표시', 
        overlay=True
    ).add_to(m)

    # 클러스터 레이어 설정
    cluster_colors = {
        'HH': 'red',
        'HL': 'transparent',
        'LH': 'transparent',
        'LL': 'transparent',
        'NS': 'transparent'
    }

    # 클러스터에 따른 스타일 설정 함수
    def style_function(feature):
        return {
            'fillColor': cluster_colors.get(feature['properties']['Cluster_Label'], 'gray'),
            'color': 'transparent',
            'weight': 1,
            'fillOpacity': 0.5
        }
    nbr_final.crs = "EPSG:4326"
    # 클러스터 레이어 추가
    cluster_layer = folium.FeatureGroup(name='전체사고 핫스팟')
    folium.GeoJson(
        nbr_final,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['acc_counts'],
            aliases=['사고수:']
        )
    ).add_to(cluster_layer)
    cluster_layer.add_to(m)

    # 안전쉼터 레이어 설정 및 추가
    shelter_layer = folium.FeatureGroup(name='안전쉼터')
    for idx, row in safety_place.iterrows():
        folium.CircleMarker(
            location=(row['위도'], row['경도']),
            popup=row['쉼터명'],
            radius=3,
            color='blue',
            fill=True,
            fill_color='green',
            fill_opacity=1
        ).add_to(shelter_layer)
    shelter_layer.add_to(m)

    # 탐방로 레이어 설정 및 추가
    trail_layer = folium.FeatureGroup(name='탐방로')
    folium.GeoJson(
        df_탐방로[df_탐방로.geometry.length > 0.001],
        name='Trails',
        style_function=lambda feature: {
            'fillColor': 'blue',
            'color': 'blue',
            'weight': 3,
            'fillOpacity': 0.4,
            'opacity': 0.4
        }
    ).add_to(trail_layer)
    trail_layer.add_to(m)

    # 위치표지판 레이어 설정 및 추가
    sign_layer = folium.FeatureGroup(name='다목적위치표지판')
    for idx, row in sign_place.iterrows():
        folium.CircleMarker(
            location=(row['위도'], row['경도']),
            popup=row['위치'],
            radius=3,
            color='orange',
            fill=True,
            fill_color='orange',
            fill_opacity=0.8
        ).add_to(sign_layer)
    sign_layer.add_to(m)

    geojson_data = json.loads(selected_npark_boundary.to_json())
    folium.GeoJson(
        geojson_data,
        name='국립공원 경계',
        style_function=lambda feature: {
            'color': 'yellow',
            'weight': 2,
            'fillOpacity': 0
        }
    ).add_to(m)


    def filter_hotspots_far_from_safetyplace(nbr_final, safety_place, min_distance=100, cluster_label='HH'):
        # 안전쉼터 데이터를 GeoDataFrame으로 변환
        safety_place_gdf = gpd.GeoDataFrame(
            safety_place,
            geometry=gpd.points_from_xy(safety_place.경도, safety_place.위도),
            crs='EPSG:4326'
        )

        # 좌표계 변환
        nbr_final_utm = nbr_final.to_crs(epsg=5174)
        safety_place_utm = safety_place_gdf.to_crs(epsg=5174)

        # 'HH' 클러스터 라벨이 지정된 핫스팟 선택
        nbr_final_핫스팟 = nbr_final_utm[nbr_final_utm['Cluster_Label'] == cluster_label]

                # nbr_final_핫스팟이 비어있는 경우 예외 발생
        if len(nbr_final_핫스팟) == 0:
            raise ValueError("No hotspots found with the specified cluster label.")


        # 각 사고 핫스팟에서 가장 가까운 안전쉼터까지의 최소 거리 계산
        def calculate_min_distance_to_AED(hotspot, shelters):
            # 사고 핫스팟과 모든 안전쉼터 사이의 거리 계산 후 최소값 반환
            distances = shelters.geometry.distance(hotspot.geometry)
            return distances.min()
        # 최소 거리 계산
        nbr_final_핫스팟['min_distance_to_shelter'] = nbr_final_핫스팟.apply(
            lambda hotspot: calculate_min_distance_to_AED(hotspot, safety_place_utm), axis=1
        )

        # 지정된 최소 거리 이상 떨어진 핫스팟 필터링
        hotspots_far_from_shelters = nbr_final_핫스팟[nbr_final_핫스팟['min_distance_to_shelter'] > min_distance]

        # WGS84 좌표계로 변환
        hotspots_far_from_shelters_wgs84 = hotspots_far_from_shelters.to_crs(epsg=4326)
        
        return hotspots_far_from_shelters_wgs84

    ########## 이격거리 조절가능 ########################## 
    out_hotspot_layer = folium.FeatureGroup(name='안전쉼터 '+str(distance)+'m 초과 핫스팟지점')
    try:
        for idx, row in filter_hotspots_far_from_safetyplace(nbr_final, safety_place, distance, 'HH').iterrows():
            folium.CircleMarker(
                location=[row.geometry.centroid.y, row.geometry.centroid.x],
                radius=3,
                color='gold',
                fill=True,
                fill_color='gold',
                fill_opacity=0.8
            ).add_to(out_hotspot_layer)
        out_hotspot_layer.add_to(m)
    except ValueError as e:
        st.error("사고 발생 건수가 적어 지도 분석이 어려워요. 다른 공원을 분석해 주세요!")

    # 레이어 컨트롤 추가하여 사용자가 레이어 선택 가능하게 함
    folium.LayerControl().add_to(m)

    return m















########################################### 핫스팟과 AED가 입력한 거리보다 넘어간 핫스팟 위치 표시 ##################################
def make_hotspot_heart(selected_national_park_accident,selected_npark_boundary,df_AED,distance=500):
    single_polygon = selected_npark_boundary['geometry'].unary_union
    # GeoJSON 형식으로 변환
    geojson_polygon = single_polygon.__geo_interface__
    resolution=9
    # 북한산 국립공원을 커버하는 H3 육각형 인덱스 생성
    hexagons = list(h3.polyfill_geojson(geojson_polygon, resolution))

    # polygon = seoul_npark_boundary.iloc[0]['geometry']
    # 육각형 경계 좌표 생성
    hexagon_polygons = [Polygon(h3.h3_to_geo_boundary(hex, geo_json=True)) for hex in hexagons]

    # GeoDataFrame 생성
    gdf_polygon = gpd.GeoDataFrame(geometry=hexagon_polygons)

    # 심장문제만 필터링
    selected_national_park_accident = selected_national_park_accident[selected_national_park_accident['유형']=='심장문제']
    prices = gpd.sjoin(gdf_polygon, selected_national_park_accident[['geometry']], op='contains')

    # 각 geometry에 대한 카운트를 계산
    nbr_avg_price = prices.groupby('geometry').size().reset_index(name='acc_counts')
    # 병합하기
    nbr_final = gdf_polygon.merge(nbr_avg_price, on='geometry', how='left')
    # 병합 결과에서 결측값을 0으로 채우기
    nbr_final['acc_counts'] = nbr_final['acc_counts'].fillna(0)
    w =  lps.weights.Queen.from_dataframe(nbr_final)
    w.transform = 'r'

    # 속성 유사성(Attribute similarity) / 가중치 적용 사고발생건수 추가
    nbr_final['weighted_acc_counts'] = lps.weights.lag_spatial(w, nbr_final['acc_counts'])

    # 광역적 공간 자기상관 / 상관계수값 출력 
    y = nbr_final.acc_counts
    moran = esda.Moran(y, w)
    moran_local = Moran_Local(y, w)
    nbr_final['Cluster'] = moran_local.q
    cluster_map = {1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}
    nbr_final['Cluster_Label'] = nbr_final['Cluster'].map(cluster_map)
    moran_local = Moran_Local(y, w)
    # p-value 기준으로 유의미한 결과만 필터링
    p_threshold = 0.05  # 유의수준 설정
    nbr_final['Significant'] = moran_local.p_sim < p_threshold

    # 유의미하지 않은 관측치에 대해 'NS' 라벨 할당
    nbr_final['Cluster_Label'] = nbr_final.apply(lambda row: 'NS' if not row['Significant'] else row['Cluster_Label'], axis=1)

    # 'Significant' 열은 이제 필요 없으므로 제거하거나, 유지하려면 이 단계를 생략
    nbr_final.drop(columns=['Significant'], inplace=True)

    center = nbr_final.geometry.centroid.unary_union.centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=12)
    # 전체 화면 버튼 추가
    fullscreen = Fullscreen(position='topleft',  # 버튼 위치
                            title='전체 화면',     # 마우스 오버시 표시될 텍스트
                            title_cancel='전체 화면 해제',  # 전체 화면 모드 해제 버튼의 텍스트
                            force_separate_button=True)  # 전체 화면 버튼을 별도의 버튼으로 표시
    m.add_child(fullscreen)
    # VWorld Satellite Layer 추가
    vworld_key = "BF677CB9-D1EA-3831-B328-084A9AE3CDCC"
    satellite_layer = folium.TileLayer(
        tiles=f"http://api.vworld.kr/req/wmts/1.0.0/{vworld_key}/Satellite/{{z}}/{{y}}/{{x}}.jpeg",
        attr='VWorld Satellite', 
        name='위성지도'
    ).add_to(m)

    # VWorld Hybrid Layer 추가
    hybrid_layer = folium.TileLayer(
        tiles=f"http://api.vworld.kr/req/wmts/1.0.0/{vworld_key}/Hybrid/{{z}}/{{y}}/{{x}}.png",
        attr='VWorld Hybrid', 
        name='지명표시', 
        overlay=True
    ).add_to(m)

    # 클러스터 레이어 설정
    cluster_colors = {
        'HH': 'red',
        'HL': 'transparent',
        'LH': 'transparent',
        'LL': 'transparent',
        'NS': 'transparent'
    }

    # 클러스터에 따른 스타일 설정 함수
    def style_function(feature):
        return {
            'fillColor': cluster_colors.get(feature['properties']['Cluster_Label'], 'gray'),
            'color': 'transparent',
            'weight': 1,
            'fillOpacity': 0.5
        }
    nbr_final.crs = "EPSG:4326"
    # 클러스터 레이어 추가
    cluster_layer = folium.FeatureGroup(name='심장사고 핫스팟')
    folium.GeoJson(
        nbr_final,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['acc_counts'],
            aliases=['사고수:']
        )
    ).add_to(cluster_layer)
    cluster_layer.add_to(m)

    # 탐방로 레이어 설정 및 추가
    trail_layer = folium.FeatureGroup(name='탐방로')
    folium.GeoJson(
        df_탐방로[df_탐방로.geometry.length > 0.001],
        name='Trails',
        style_function=lambda feature: {
            'fillColor': 'blue',
            'color': 'blue',
            'weight': 3,
            'fillOpacity': 0.4,
            'opacity': 0.4
        }
    ).add_to(trail_layer)
    trail_layer.add_to(m)

    # 위치표지판 레이어 설정 및 추가
    sign_layer = folium.FeatureGroup(name='다목적위치표지판')
    for idx, row in sign_place.iterrows():
        folium.CircleMarker(
            location=(row['위도'], row['경도']),
            popup=row['위치'],
            radius=3,
            color='orange',
            fill=True,
            fill_color='orange',
            fill_opacity=0.8
        ).add_to(sign_layer)
    sign_layer.add_to(m)

    geojson_data = json.loads(selected_npark_boundary.to_json())
    folium.GeoJson(
        geojson_data,
        name='국립공원 경계',
        style_function=lambda feature: {
            'color': 'yellow',
            'weight': 2,
            'fillOpacity': 0
        }
    ).add_to(m)

    # AED 위치
    df_AED_layer = folium.FeatureGroup(name='AED')
    for idx, row in df_AED.iterrows():
        folium.CircleMarker(
            location=(row['위도'], row['경도']),
            popup=row['명칭'],
            radius=3,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=1
        ).add_to(df_AED_layer)
    df_AED_layer.add_to(m)


    # 심장사고지점 추가
    seoul_accident_fall_layer = folium.FeatureGroup(name='심장사고 사고지점')
    for idx, row in selected_national_park_accident.iterrows():
        folium.CircleMarker(
            location=(row['위도_변환'], row['경도_변환']),
            popup=row['사고장소'],
            radius=3,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=1
        ).add_to(seoul_accident_fall_layer)
    seoul_accident_fall_layer.add_to(m)


    def filter_hotspots_far_from_AED(nbr_final, df_AED, min_distance=100, cluster_label='HH'):
        # 안전쉼터 데이터를 GeoDataFrame으로 변환
        df_AED_gdf = gpd.GeoDataFrame(
            df_AED,
            geometry=gpd.points_from_xy(df_AED.경도, df_AED.위도),
            crs='EPSG:4326'
        )

        # 좌표계 변환
        nbr_final_utm = nbr_final.to_crs(epsg=5174)
        df_AED_utm = df_AED_gdf.to_crs(epsg=5174)

        # 'HH' 클러스터 라벨이 지정된 핫스팟 선택
        nbr_final_핫스팟 = nbr_final_utm[nbr_final_utm['Cluster_Label'] == cluster_label]
        
        # nbr_final_핫스팟이 비어있는 경우 예외 발생
        if len(nbr_final_핫스팟) == 0:
            raise ValueError("No hotspots found with the specified cluster label.")

        # 각 사고 핫스팟에서 가장 가까운 안전쉼터까지의 최소 거리 계산
        def calculate_min_distance_to_shelter(hotspot, shelters):
            # 사고 핫스팟과 모든 안전쉼터 사이의 거리 계산 후 최소값 반환
            distances = shelters.geometry.distance(hotspot.geometry)
            return distances.min()

        # 최소 거리 계산
        nbr_final_핫스팟['min_distance_to_shelter'] = nbr_final_핫스팟.apply(
            lambda hotspot: calculate_min_distance_to_shelter(hotspot, df_AED_utm), axis=1
        )

        # 지정된 최소 거리 이상 떨어진 핫스팟 필터링
        hotspots_far_from_shelters = nbr_final_핫스팟[nbr_final_핫스팟['min_distance_to_shelter'] > min_distance]

        # WGS84 좌표계로 변환
        hotspots_far_from_shelters_wgs84 = hotspots_far_from_shelters.to_crs(epsg=4326)
        
        return hotspots_far_from_shelters_wgs84

    ########## 이격거리 조절가능 ########################## 
    out_hotspot_layer = folium.FeatureGroup(name='AED '+str(distance)+'m 초과 핫스팟지점')
    try:
        for idx, row in filter_hotspots_far_from_AED(nbr_final, df_AED, distance, 'HH').iterrows():
            folium.CircleMarker(
                location=[row.geometry.centroid.y, row.geometry.centroid.x],
                radius=3,
                color='gold',
                fill=True,
                fill_color='gold',
                fill_opacity=0.8
            ).add_to(out_hotspot_layer)
        out_hotspot_layer.add_to(m)
    except ValueError as e:
        st.error("심장 사고 발생 건수가 적어 지도 분석이 어려워요. 다른 공원을 분석해 주세요! ")

    # 레이어 컨트롤 추가하여 사용자가 레이어 선택 가능하게 함
    folium.LayerControl().add_to(m)

    return m
















########################################### 핫스팟과 추락위험지역이 입력한 거리보다 넘어간 핫스팟 위치 표시 ##################################
def make_hotspot_fall(selected_national_park_accident,selected_npark_boundary,df_fall,distance=500):
    single_polygon = selected_npark_boundary['geometry'].unary_union
    # GeoJSON 형식으로 변환
    geojson_polygon = single_polygon.__geo_interface__
    resolution=9
    # 북한산 국립공원을 커버하는 H3 육각형 인덱스 생성
    hexagons = list(h3.polyfill_geojson(geojson_polygon, resolution))

    # polygon = seoul_npark_boundary.iloc[0]['geometry']
    # 육각형 경계 좌표 생성
    hexagon_polygons = [Polygon(h3.h3_to_geo_boundary(hex, geo_json=True)) for hex in hexagons]

    # GeoDataFrame 생성
    gdf_polygon = gpd.GeoDataFrame(geometry=hexagon_polygons)

    # 심장문제만 필터링
    selected_national_park_accident = selected_national_park_accident[selected_national_park_accident['유형']=='추락']
    prices = gpd.sjoin(gdf_polygon, selected_national_park_accident[['geometry']], op='contains')

    # 각 geometry에 대한 카운트를 계산
    nbr_avg_price = prices.groupby('geometry').size().reset_index(name='acc_counts')
    # 병합하기
    nbr_final = gdf_polygon.merge(nbr_avg_price, on='geometry', how='left')
    # 병합 결과에서 결측값을 0으로 채우기
    nbr_final['acc_counts'] = nbr_final['acc_counts'].fillna(0)
    w =  lps.weights.Queen.from_dataframe(nbr_final)
    w.transform = 'r'

    # 속성 유사성(Attribute similarity) / 가중치 적용 사고발생건수 추가
    nbr_final['weighted_acc_counts'] = lps.weights.lag_spatial(w, nbr_final['acc_counts'])

    # 광역적 공간 자기상관 / 상관계수값 출력 
    y = nbr_final.acc_counts
    moran = esda.Moran(y, w)
    moran_local = Moran_Local(y, w)
    nbr_final['Cluster'] = moran_local.q
    cluster_map = {1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}
    nbr_final['Cluster_Label'] = nbr_final['Cluster'].map(cluster_map)
    moran_local = Moran_Local(y, w)
    # p-value 기준으로 유의미한 결과만 필터링
    p_threshold = 0.05  # 유의수준 설정
    nbr_final['Significant'] = moran_local.p_sim < p_threshold

    # 유의미하지 않은 관측치에 대해 'NS' 라벨 할당
    nbr_final['Cluster_Label'] = nbr_final.apply(lambda row: 'NS' if not row['Significant'] else row['Cluster_Label'], axis=1)

    # 'Significant' 열은 이제 필요 없으므로 제거하거나, 유지하려면 이 단계를 생략
    nbr_final.drop(columns=['Significant'], inplace=True)

    center = nbr_final.geometry.centroid.unary_union.centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=12)
    # 전체 화면 버튼 추가
    fullscreen = Fullscreen(position='topleft',  # 버튼 위치
                            title='전체 화면',     # 마우스 오버시 표시될 텍스트
                            title_cancel='전체 화면 해제',  # 전체 화면 모드 해제 버튼의 텍스트
                            force_separate_button=True)  # 전체 화면 버튼을 별도의 버튼으로 표시
    m.add_child(fullscreen)
    # VWorld Satellite Layer 추가
    vworld_key = "BF677CB9-D1EA-3831-B328-084A9AE3CDCC"
    satellite_layer = folium.TileLayer(
        tiles=f"http://api.vworld.kr/req/wmts/1.0.0/{vworld_key}/Satellite/{{z}}/{{y}}/{{x}}.jpeg",
        attr='VWorld Satellite', 
        name='위성지도'
    ).add_to(m)

    # VWorld Hybrid Layer 추가
    hybrid_layer = folium.TileLayer(
        tiles=f"http://api.vworld.kr/req/wmts/1.0.0/{vworld_key}/Hybrid/{{z}}/{{y}}/{{x}}.png",
        attr='VWorld Hybrid', 
        name='지명표시', 
        overlay=True
    ).add_to(m)

    # 클러스터 레이어 설정
    cluster_colors = {
        'HH': 'red',
        'HL': 'transparent',
        'LH': 'transparent',
        'LL': 'transparent',
        'NS': 'transparent'
    }

    # 클러스터에 따른 스타일 설정 함수
    def style_function(feature):
        return {
            'fillColor': cluster_colors.get(feature['properties']['Cluster_Label'], 'gray'),
            'color': 'transparent',
            'weight': 1,
            'fillOpacity': 0.5
        }
    nbr_final.crs = "EPSG:4326"
    # 클러스터 레이어 추가
    cluster_layer = folium.FeatureGroup(name='추락사고 핫스팟')
    folium.GeoJson(
        nbr_final,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['acc_counts'],
            aliases=['사고수:']
        )
    ).add_to(cluster_layer)
    cluster_layer.add_to(m)

    # 탐방로 레이어 설정 및 추가
    trail_layer = folium.FeatureGroup(name='탐방로')
    folium.GeoJson(
        df_탐방로[df_탐방로.geometry.length > 0.001],
        name='Trails',
        style_function=lambda feature: {
            'fillColor': 'blue',
            'color': 'blue',
            'weight': 3,
            'fillOpacity': 0.4,
            'opacity': 0.4
        }
    ).add_to(trail_layer)
    trail_layer.add_to(m)

    # 위치표지판 레이어 설정 및 추가
    sign_layer = folium.FeatureGroup(name='다목적위치표지판')
    for idx, row in sign_place.iterrows():
        folium.CircleMarker(
            location=(row['위도'], row['경도']),
            popup=row['위치'],
            radius=3,
            color='orange',
            fill=True,
            fill_color='orange',
            fill_opacity=0.8
        ).add_to(sign_layer)
    sign_layer.add_to(m)

    geojson_data = json.loads(selected_npark_boundary.to_json())
    folium.GeoJson(
        geojson_data,
        name='국립공원 경계',
        style_function=lambda feature: {
            'color': 'yellow',
            'weight': 2,
            'fillOpacity': 0
        }
    ).add_to(m)


    # 추락위험지역 설정 및 추가
    fall_spot_layer = folium.FeatureGroup(name='기존 추락위험지역')
    for idx, row in df_fall.iterrows():
        folium.CircleMarker(
            location=(row['위도'], row['경도']),
            popup=row['세부위치'],
            radius=3,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=1
        ).add_to(fall_spot_layer)
    fall_spot_layer.add_to(m)

    # 추락사 사고지점 추가
    seoul_accident_fall_layer = folium.FeatureGroup(name='추락사 사고지점')
    for idx, row in selected_national_park_accident.iterrows():
        folium.CircleMarker(
            location=(row['위도_변환'], row['경도_변환']),
            popup=row['사고장소'],
            radius=3,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=1
        ).add_to(seoul_accident_fall_layer)
    seoul_accident_fall_layer.add_to(m)



    def filter_hotspots_far_from_fall(nbr_final, df_fall, min_distance=100, cluster_label='HH'):
        # 안전쉼터 데이터를 GeoDataFrame으로 변환
        df_fall_gdf = gpd.GeoDataFrame(
            df_fall,
            geometry=gpd.points_from_xy(df_fall.경도, df_fall.위도),
            crs='EPSG:4326'
        )

        # 좌표계 변환
        nbr_final_utm = nbr_final.to_crs(epsg=5174)
        df_fall_utm = df_fall_gdf.to_crs(epsg=5174)

        # 'HH' 클러스터 라벨이 지정된 핫스팟 선택
        nbr_final_핫스팟 = nbr_final_utm[nbr_final_utm['Cluster_Label'] == cluster_label]

        # nbr_final_핫스팟이 비어있는 경우 예외 발생
        if len(nbr_final_핫스팟) == 0:
            raise ValueError("No hotspots found with the specified cluster label.")

        # 각 사고 핫스팟에서 가장 가까운 안전쉼터까지의 최소 거리 계산
        def calculate_min_distance_to_fall(hotspot, shelters):
            # 사고 핫스팟과 모든 안전쉼터 사이의 거리 계산 후 최소값 반환
            distances = shelters.geometry.distance(hotspot.geometry)
            return distances.min()
        # 최소 거리 계산
        nbr_final_핫스팟['min_distance_to_shelter'] = nbr_final_핫스팟.apply(
            lambda hotspot: calculate_min_distance_to_fall(hotspot, df_fall_utm), axis=1
        )

        # 지정된 최소 거리 이상 떨어진 핫스팟 필터링
        hotspots_far_from_shelters = nbr_final_핫스팟[nbr_final_핫스팟['min_distance_to_shelter'] > min_distance]

        # WGS84 좌표계로 변환
        hotspots_far_from_shelters_wgs84 = hotspots_far_from_shelters.to_crs(epsg=4326)
        
        return hotspots_far_from_shelters_wgs84

    ########## 이격거리 조절가능 ########################## 
    out_hotspot_layer = folium.FeatureGroup(name='추락위험지역 '+str(distance)+'m 초과 핫스팟지점')
    try:
        for idx, row in filter_hotspots_far_from_fall(nbr_final, df_fall, distance, 'HH').iterrows():
            folium.CircleMarker(
                location=[row.geometry.centroid.y, row.geometry.centroid.x],
                radius=3,
                color='gold',
                fill=True,
                fill_color='gold',
                fill_opacity=0.8
            ).add_to(out_hotspot_layer)
        out_hotspot_layer.add_to(m)
    except ValueError as e:
        st.error("추락사고 발생 건수가 적어 지도 분석이 어려워요. 다른 공원을 분석해 주세요! ")

    # 레이어 컨트롤 추가하여 사용자가 레이어 선택 가능하게 함
    folium.LayerControl().add_to(m)

    return m









# Heat map
def make_heatmap(selected_national_park_accident,selected_npark_boundary):
    m = v_world(selected_national_park_accident)
    fullscreen = Fullscreen(position='topleft',  # 버튼 위치
                        title='전체 화면',     # 마우스 오버시 표시될 텍스트
                        title_cancel='전체 화면 해제',  # 전체 화면 모드 해제 버튼의 텍스트
                        force_separate_button=True)  # 전체 화면 버튼을 별도의 버튼으로 표시
    m.add_child(fullscreen)

    # 탐방로 레이어 설정 및 추가
    trail_layer = folium.FeatureGroup(name='탐방로')
    folium.GeoJson(
        df_탐방로[df_탐방로.geometry.length > 0.001],
        name='Trails',
        style_function=lambda feature: {
            'fillColor': 'blue',
            'color': 'blue',
            'weight': 3,
            'fillOpacity': 0.4,
            'opacity': 0.4
        }
    ).add_to(trail_layer)
    trail_layer.add_to(m)

    # seoul_npark_boundary GeoDataFrame을 GeoJson으로 변환 및 추가
    geojson_data = json.loads(selected_npark_boundary.to_json())
    folium.GeoJson(
        geojson_data,
        name='경계',
        style_function=lambda feature: {
            'color': 'yellow',
            'weight': 2,
            'fillOpacity': 0
        }
    ).add_to(m)

    # 사고 위치 데이터 준비 (위도, 경도)
    accident_locations = selected_national_park_accident[['위도_변환', '경도_변환']].values.tolist()

    # 히트맵 레이어 생성
    heat_map = plugins.HeatMap(accident_locations, radius=15, gradient={0.2: 'blue', 0.4: 'green', 0.6: 'yellow', 0.8: 'orange', 1: 'red'})

    # 레이어 그룹 생성 및 히트맵 레이어 추가
    layer_group = folium.FeatureGroup().add_child(heat_map)

    # 레이어 그룹을 지도 객체에 추가
    m.add_child(layer_group)

    # 레이어 컨트롤 추가
    folium.LayerControl().add_to(m)

    return m

def plot_donut_chart(df):
    # value_counts를 사용해 각 카테고리의 빈도수 계산
    value_counts = df['유형'].value_counts()

    # 도넛 차트를 위한 데이터와 레이아웃 설정
    fig = go.Figure(data=[go.Pie(labels=value_counts.index, values=value_counts, hole=.3)])
    
    # 차트 제목 및 레이아웃 설정
    fig.update_layout(
        title_text='사고 유형 분포',
        # 글씨 크기 조정
        title_font_size=20,
        legend_title_font_size=12,
        font=dict(size=18)
    )

    return fig

#######################
# Sidebar
with st.sidebar:
    st.title('국립공원 Dashboard')
    nationpark_list = ['북한산', '설악산', '지리산', '무등산', '덕유산', '계룡산', '월출산', '태백산', '월악산', '내장산',
       '속리산', '주왕산', '소백산', '변산반도', '치악산', '오대산', '가야산', '다도해해상', '한려해상', '경주',
       '태안해안']
    st.selectbox('국립공원 선택', nationpark_list,key='selected_national_park')
    year_list = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

    year_list.insert(0,'전체')
    year = st.multiselect('연도 선택',year_list,key='year',default='전체')
    gender_list = ['전체','남','여']
    st.selectbox('성별 선택',gender_list,key='gender')
    month = st.multiselect('월별 선택',
    ['전체','1월', '2월','3월','4월','5월','6월','7월','8월','9월','10월','11월','12월'],key='month',default='전체')
    age = st.multiselect('연령대 선택',
    ['전체','20대미만','20대', '30대','40대','50대', '60대', '70대 이상', '미상', '집단'],key='age',default='전체')
    resolution = st.slider('기존 안전시설물-사고 핫스팟 이격거리 설정', 100, 1000, 500,100,key='distance')
    st.write('핫스팟에서 벗어난 기존 설치 지점이 곧 핫스팟 내 안전시설물 우선설치 필요 지점 예측을 말해요.')
    button = st.button('분석 시작')
    image1 = './logo/국공.svg'
    image2 = './logo/Bigleader.png'
    st.write('')
    st.write('')
    st.image(image1)
    st.image(image2)


# Dashboard Main Panel
if not button:
    # custom_html = """
    # <div class="banner">
    #     <img src="" alt="Banner Image">
    # </div>
    # <style>
    #     .banner {
    #         width: 160%;
    #         height: 200px;
    #         overflow: hidden;
    #     }
    #     .banner img {
    #         width: 100%;
    #         object-fit: cover;
    #     }
    # </style>
    # """
    # # Display the custom HTML
    # st.components.v1.html(custom_html)
    st.markdown("""
    <style>
    body {
    background-size: cover;
    background-attachment: fixed; /* 배경 이미지 고정 */
    }

    .stApp { /* Streamlit 앱의 최대 너비 조정 */
        margin: auto;
    }

    .title {
        font-size: 48px; /* 제목 크기 조정 */
        font-weight: bold;
        text-align: center;
        margin-top: 30px;
        color: #6A877F; /* 부제목 색상 */
        text-shadow: 2px 2px 4px #000000; /* 제목에 그림자 효과 추가 */
    }
    .subtitle {
        font-size: 28px; /* 부제목 크기 조정 */
        text-align: center;
        margin-bottom: 30px;
        color: #6A877F; /* 부제목 색상 */
        text-shadow: 1px 1px 2px #000000; /* 부제목에 그림자 효과 추가 */
    }
    .content {
        font-family: 'Noto Sans KR', sans-serif; /* 본문 글씨체 변경 */
        font-size: 30px; /* 본문 글씨 크기 변경 */
        padding: 20px;
        background-color: rgba(255,255,255,0.8); /* 본문 배경색 추가 및 투명도 조정 */
        border-radius: 15px; /* 본문 모서리 둥글게 */
        box-shadow: 0 4px 8px rgba(0,0,0,0.1); /* 본문에 그림자 효과 추가 */
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    # 페이지 타이틀
    st.markdown('<h1 class="title">국립공원 안전사고 분석 리포트</h1>', unsafe_allow_html=True)

    # 페이지 부제목 및 소개
    st.markdown('<h2 class="subtitle">- 안전사고를 줄이기 위한 분석 및 대책 지원 -</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="content">
        <p>국립공원 내 안전사고를 분석하고 효과적인 예방대책을 마련하는 페이지입니다. 
왼쪽 사이드바에는 사고 패턴을 파악할 수 있도록 다양한 시각화 도구와 분석 결과를 제공합니다. </p>
        <p>좌측 사이드바를 클릭하여 분석 자료를 확인하세요.</p>
    </div>
    """, unsafe_allow_html=True)
   # CSS 스타일
if button:
    with st.spinner('Wait for it...'):
        # npark_boundary = gpd.read_file('./data/Protected_areas_OECM_Republic_of_Korea_ver_2023.shp', encoding='cp949')
        # park_data = pd.read_csv('./data/240301_final_data_ver2.csv')
        # safety_place = pd.read_csv('./data/안전쉼터_final.csv')
        # sign_place = pd.read_excel('./data/북한산 다목적 위치표지판 현황.xlsx')
        # df_탐방로 = gpd.read_file('./data/국립공원시설_선형시설.shp')
        # #######################

        # gdf_park_data = gpd.GeoDataFrame(park_data, 
        #                             geometry=gpd.points_from_xy(park_data.경도_변환, park_data.위도_변환),
        #                             crs='epsg:4326'
        #                             )
        # gdf_safety_place = gpd.GeoDataFrame(safety_place, 
        #                             geometry=gpd.points_from_xy(safety_place.경도, safety_place.위도),
        #                             crs='epsg:4326'
        #                             )
        # gdf_sign_place = gpd.GeoDataFrame(sign_place, 
        #                             geometry=gpd.points_from_xy(sign_place.경도, sign_place.위도),
        #                             crs='epsg:4326'
        #                             )
        
        # GeoPackage 파일로부터 GeoDataFrame 불러오기
        plt.rcParams['font.family'] ='Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] =False
        gdf_park_data = gpd.read_file("./data/park_data2.gpkg", layer='park_data')
        sign_place = gpd.read_file("./data/sign_place.gpkg", layer='sign_place')
        df_탐방로 = gpd.read_file('./data/국립공원시설_선형시설.shp')
        npark_boundary = gpd.read_file('./data/npark_boundary.gpkg',layer='npark_boundary')
        df_AED = pd.read_csv('./data/AED_final.csv')
        df_fall = pd.read_csv('./data/추락위험지역_final.csv')
        safety_place = pd.read_csv("./data/안전쉼터_final.csv")


        # #######################
        # npark_boundary = gpd.read_file('./data/npark_boundary.gpkg',layer='npark_boundary')
        # gdf_park_data = gpd.read_file("./data/park_data.gpkg", layer='park_data')
        # safety_place = gpd.read_file("./data/safety_place.gpkg", layer='safety_place')
        # sign_place = gpd.read_file("./data/sign_place.gpkg", layer='sign_place')
        
        selected_national_park = st.session_state['selected_national_park']
        safety_place = safety_place[safety_place['국립공원명']==selected_national_park]
        df_AED = df_AED[df_AED['국립공원명']==selected_national_park]
        df_fall = df_fall[df_fall['국립공원명']==selected_national_park]
        selected_npark_boundary = find_boundary(npark_boundary,selected_national_park)
        selected_npark_boundary_hotspot = find_boundary_hotspot(npark_boundary,selected_national_park)
        selected_national_park_accident = sjoin(gdf_park_data,selected_npark_boundary,selected_national_park)
        selected_national_park_accident_hotspot = sjoin(gdf_park_data,selected_npark_boundary_hotspot,selected_national_park)
    
        if '전체' not in st.session_state['year']:
             selected_national_park_accident = selected_national_park_accident[selected_national_park_accident['연도'].isin(st.session_state['year'])]
             selected_national_park_accident_hotspot = selected_national_park_accident_hotspot[selected_national_park_accident_hotspot['연도'].isin(st.session_state['year'])]
        if st.session_state['gender']!='전체':
            selected_national_park_accident = selected_national_park_accident[selected_national_park_accident['성별']==st.session_state['gender']]
            selected_national_park_accident_hotspot = selected_national_park_accident_hotspot[selected_national_park_accident_hotspot['성별']==st.session_state['gender']]
        if '전체' not in st.session_state['month']:
             selected_national_park_accident = selected_national_park_accident[selected_national_park_accident['월'].isin(st.session_state['month'])]
             selected_national_park_accident_hotspot = selected_national_park_accident_hotspot[selected_national_park_accident_hotspot['월'].isin(st.session_state['month'])]
        if '전체' not in st.session_state['age']:
            selected_national_park_accident = selected_national_park_accident[selected_national_park_accident['연령대'].isin(st.session_state['age'])]
            selected_national_park_accident_hotspot = selected_national_park_accident_hotspot[selected_national_park_accident_hotspot['연령대'].isin(st.session_state['age'])]
        # try:
        map_center_lat,map_center_lon = find_center_latitudes_longtitudes(selected_national_park_accident)
        
        vworld_key="BF677CB9-D1EA-3831-B328-084A9AE3CDCC" # VWorld API key
        layer = "Satellite" # VWorld layer
        tileType = "jpeg" # tile type
        accident_list = selected_national_park_accident['사고원인그룹'].value_counts().index

        col = st.columns((2.5, 5.5), gap='medium')

        with col[0]:
            st.metric(label="사고 건수", value=len(selected_national_park_accident))
            fig1 = plot_donut_chart(selected_national_park_accident)
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("""
                <div class="content">
                    <p>“ 차트 활용법 <br>
                    1. 차트의 경우 오른쪽 상단에 마우스를 올려둘 시 전체화면으로 확대 버튼이 떠요. <br>
                    2. 차트 클릭시 인터렉티브하게 반응해요.(사고 건수 파악 가능)  ” </p>
                </div>
                """, unsafe_allow_html=True)

        with col[1]:
            st.markdown('#### 사고 현황판')
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["사고 현황", "전체 사고 히트맵","안전쉼터위치 선정","AED위치 선정", "추락위험지역 선정"])
            with tab1:
                # 지도 생성
                m = make_pointplot(selected_national_park_accident,selected_npark_boundary)
                folium_static(m)

            with tab2:
                # 지도 생성
                m2 = make_heatmap(selected_national_park_accident,selected_npark_boundary)
                folium_static(m2)

            with tab3:
                # 지도 생성
                m3 = make_hotspot_safetyplace(selected_national_park_accident_hotspot,selected_npark_boundary_hotspot,safety_place,st.session_state['distance'])
                folium_static(m3)

            with tab4:
                # 지도 생성
                m4 = make_hotspot_heart(selected_national_park_accident_hotspot,selected_npark_boundary_hotspot,df_AED,st.session_state['distance'])
                folium_static(m4)

            with tab5:
                # 지도 생성
                m5 = make_hotspot_fall(selected_national_park_accident_hotspot,selected_npark_boundary_hotspot,df_fall,st.session_state['distance'])
                folium_static(m5)

