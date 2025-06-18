"""
Generador de Árbol Binario Limpio y Profesional
==================================================

Este script crea visualizaciones de árboles de decisión limpias, sin superposiciones
y con un diseño profesional para el análisis de Hepatitis B.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.tree import DecisionTreeClassifier
from pycaret.classification import load_model
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class CleanBinaryTreeVisualizer:
    def __init__(self, model_path, data_path):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.data = None
        self.tree_model = None
        
    def load_data_and_model(self):
        """Cargar datos y modelo"""
        print("📁 Cargando datos y modelo...")
        
        # Cargar datos
        self.data = pd.read_csv(self.data_path)
        self.feature_names = [col for col in self.data.columns if col != 'Hepatitis_B']
        self.target_name = 'Hepatitis_B'
        
        # Cargar modelo
        self.model = load_model(self.model_path)
        
        print(f"✅ Datos cargados: {self.data.shape}")
        print(f"✅ Features: {len(self.feature_names)}")
        
    def create_clean_decision_tree(self):
        """Crear un árbol de decisión limpio y simple"""
        print("🌳 Creando árbol de decisión simplificado...")
        
        X = self.data[self.feature_names]
        y = self.data[self.target_name]
        
        # Crear árbol simple y limpio
        self.tree_model = DecisionTreeClassifier(
            max_depth=4,  # Profundidad limitada para evitar amontonamientos
            min_samples_split=100,  # Nodos más grandes
            min_samples_leaf=50,    # Hojas más grandes
            criterion='gini',
            random_state=42
        )
        
        self.tree_model.fit(X, y)
        
        # Calcular precisión
        tree_pred = self.tree_model.predict(X)
        original_pred = self.model.predict(self.data.drop(columns=[self.target_name]))
        
        tree_accuracy = accuracy_score(y, tree_pred)
        original_accuracy = accuracy_score(y, original_pred)
        
        print(f"📊 Precisión del árbol: {tree_accuracy:.3f}")
        print(f"📊 Precisión del modelo original: {original_accuracy:.3f}")
        
        return tree_accuracy, original_accuracy
        
    def visualize_professional_tree(self, save_path=None):
        """Crear visualización profesional del árbol"""
        print("🎨 Generando visualización profesional...")
        
        # Configurar figura con tamaño óptimo
        fig, ax = plt.subplots(1, 1, figsize=(24, 16))
        fig.patch.set_facecolor('white')
        
        # Obtener estructura del árbol
        tree = self.tree_model.tree_
        
        # Colores profesionales
        colors = {
            'decision': '#4A90E2',      # Azul para nodos de decisión
            'positive': '#7ED321',      # Verde para predicciones positivas
            'negative': '#F5A623',      # Naranja para predicciones negativas
            'edge_yes': '#7ED321',      # Verde para "Sí"
            'edge_no': '#F5A623'        # Naranja para "No"
        }
        
        # Definir posiciones para evitar superposiciones
        positions = self._calculate_optimal_positions(tree)
        
        # Dibujar el árbol
        self._draw_tree_recursive(ax, tree, 0, positions, colors)
        
        # Configurar el plot
        ax.set_xlim(-1, 25)
        ax.set_ylim(-1, 13)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Título principal
        tree_acc, orig_acc = self.create_clean_decision_tree()
        plt.suptitle(
            f'Árbol de Decisión Binario - Predicción de Hepatitis B\n'
            f'Precisión: {tree_acc:.3f} | Modelo Original: {orig_acc:.3f} | Muestras: {len(self.data)}',
            fontsize=20, fontweight='bold', y=0.95
        )
        
        # Leyenda simple y limpia
        legend_elements = [
            patches.Patch(color=colors['decision'], label='Nodo de Decisión'),
            patches.Patch(color=colors['positive'], label='Hepatitis B Positivo'),
            patches.Patch(color=colors['negative'], label='Hepatitis B Negativo')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(0.98, 0.98), fontsize=14,
                 frameon=True, fancybox=True, shadow=True)
        
        # Ajustar layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.05, left=0.02, right=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', pad_inches=0.2)
            print(f"💾 Árbol guardado en: {save_path}")
        
        plt.show()
        
    def _calculate_optimal_positions(self, tree):
        """Calcular posiciones óptimas para evitar superposiciones"""
        positions = {}
        
        # Nivel 0 (raíz)
        positions[0] = (12, 11)
        
        # Nivel 1
        if tree.children_left[0] != -1:
            positions[tree.children_left[0]] = (6, 8.5)
        if tree.children_right[0] != -1:
            positions[tree.children_right[0]] = (18, 8.5)
        
        # Nivel 2
        level1_nodes = [tree.children_left[0], tree.children_right[0]]
        level2_positions = [3, 9, 15, 21]
        pos_idx = 0
        
        for node in level1_nodes:
            if node != -1:
                if tree.children_left[node] != -1:
                    positions[tree.children_left[node]] = (level2_positions[pos_idx], 6)
                    pos_idx += 1
                if tree.children_right[node] != -1:
                    positions[tree.children_right[node]] = (level2_positions[pos_idx], 6)
                    pos_idx += 1
        
        # Nivel 3 (hojas finales)
        level3_positions = [1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5]
        pos_idx = 0
        
        # Obtener todos los nodos del nivel 2
        level2_nodes = []
        for node in level1_nodes:
            if node != -1:
                if tree.children_left[node] != -1:
                    level2_nodes.append(tree.children_left[node])
                if tree.children_right[node] != -1:
                    level2_nodes.append(tree.children_right[node])
        
        for node in level2_nodes:
            if node != -1:
                if tree.children_left[node] != -1:
                    positions[tree.children_left[node]] = (level3_positions[pos_idx], 3.5)
                    pos_idx += 1
                if tree.children_right[node] != -1:
                    positions[tree.children_right[node]] = (level3_positions[pos_idx], 3.5)
                    pos_idx += 1
        
        return positions
        
    def _draw_tree_recursive(self, ax, tree, node_id, positions, colors):
        """Dibujar el árbol recursivamente"""
        if node_id not in positions:
            return
            
        x, y = positions[node_id]
        
        # Determinar si es hoja
        is_leaf = (tree.children_left[node_id] == tree.children_right[node_id])
        
        if is_leaf:
            # Nodo hoja - predicción final
            samples = tree.n_node_samples[node_id]
            value = tree.value[node_id][0]
            
            # Determinar predicción
            pred_positive = value[1] > value[0]
            confidence = max(value) / sum(value)
            
            # Color según predicción
            color = colors['positive'] if pred_positive else colors['negative']
            prediction_text = "Hepatitis B\nPOSITIVO" if pred_positive else "Hepatitis B\nNEGATIVO"
            
            # Dibujar nodo hoja
            self._draw_leaf_node(ax, x, y, prediction_text, confidence, samples, color)
            
        else:
            # Nodo de decisión
            feature_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            samples = tree.n_node_samples[node_id]
            
            feature_name = self.feature_names[feature_idx]
            
            # Texto del nodo
            node_text = f"{feature_name}\n≤ {threshold:.2f}\nn = {samples}"
            
            # Dibujar nodo de decisión
            self._draw_decision_node(ax, x, y, node_text, colors['decision'])
            
            # Dibujar conexiones a hijos
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]
            
            if left_child in positions:
                left_x, left_y = positions[left_child]
                self._draw_edge(ax, x, y, left_x, left_y, "SÍ", colors['edge_yes'])
                self._draw_tree_recursive(ax, tree, left_child, positions, colors)
                
            if right_child in positions:
                right_x, right_y = positions[right_child]
                self._draw_edge(ax, x, y, right_x, right_y, "NO", colors['edge_no'])
                self._draw_tree_recursive(ax, tree, right_child, positions, colors)
                
    def _draw_decision_node(self, ax, x, y, text, color):
        """Dibujar nodo de decisión"""
        rect = patches.FancyBboxPatch(
            (x - 1.2, y - 0.6), 2.4, 1.2,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=2,
            alpha=0.9
        )
        ax.add_patch(rect)
        
        ax.text(x, y, text, ha='center', va='center', 
               fontsize=11, fontweight='bold', color='white')
        
    def _draw_leaf_node(self, ax, x, y, prediction, confidence, samples, color):
        """Dibujar nodo hoja"""
        rect = patches.FancyBboxPatch(
            (x - 1.0, y - 0.5), 2.0, 1.0,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=2,
            alpha=0.9
        )
        ax.add_patch(rect)
        
        # Texto principal
        ax.text(x, y + 0.1, prediction, ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')
        
        # Información adicional
        info_text = f"Conf: {confidence:.2f}\nMuestras: {samples}"
        ax.text(x, y - 0.25, info_text, ha='center', va='center',
               fontsize=8, color='white')
        
    def _draw_edge(self, ax, x1, y1, x2, y2, label, color):
        """Dibujar conexión entre nodos"""
        # Línea
        ax.plot([x1, x2], [y1 - 0.6, y2 + 0.6], color=color, 
               linewidth=3, alpha=0.8)
        
        # Etiqueta
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y, label, ha='center', va='center',
               fontsize=10, fontweight='bold', color=color,
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                        edgecolor=color, alpha=0.9))


def main():
    """Función principal"""
    print("🌳 GENERADOR DE ÁRBOL BINARIO LIMPIO")
    print("=" * 50)
    
    # Configuración
    project_root = r"c:\Users\bntmm\Desktop\GRIFOLS\BioPredict-Data-Driven-Modeling-of-Serological-Markers"
    
    # Rutas
    model_path = os.path.join(project_root, "outputs", "run_20250609_162141", "models", "best_model")
    data_path = os.path.join(project_root, "data", "processed", "mapped_target_data.csv")
    save_dir = os.path.join(project_root, "outputs", "run_20250609_162141", "clean_trees")


    # Crear directorio si no existe
    os.makedirs(save_dir, exist_ok=True)
    
    # Inicializar visualizador
    visualizer = CleanBinaryTreeVisualizer(model_path, data_path)
    
    try:
        # Cargar datos y modelo
        visualizer.load_data_and_model()
        
        # Crear y mostrar árbol
        tree_acc, orig_acc = visualizer.create_clean_decision_tree()
        
        # Generar visualización
        save_path = os.path.join(save_dir, "arbol_binario_limpio.png")
        visualizer.visualize_professional_tree(save_path)
        
        print(f"\n✅ ¡Árbol binario limpio generado exitosamente!")
        print(f"📁 Guardado en: {save_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
