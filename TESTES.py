#eduardo neves
import time
from collections import deque
import matplotlib.pyplot as plt
import heapq

class Grafos:
    def __init__(self, representacao, direcionado, ponderado):
        self.representacao = representacao
        self.direcionado = direcionado
        self.ponderado = ponderado
        self.lista_vertices = []
        if self.representacao == 'matriz':
            self.matriz = []
        else:
            self.dict_vizinhos = {}

    def __str__(self):
        ret = f'Vertices: {self.lista_vertices}\n'
        ret += 'Arestas: \n'
        
        if self.representacao == 'matriz':
            header_row = "\t".join(self.lista_vertices)
            ret += f'\t{header_row}\n'
            
            for i, linha in enumerate(self.matriz):
                vertice_nome = self.lista_vertices[i]
                linha_str = "\t".join(map(str, linha))
                ret += f'{vertice_nome}\t{linha_str}\n'
        else:
            for vertice, vizinhos in self.dict_vizinhos.items():
                ret += f'{vertice} -> {vizinhos}\n'
        return ret

    def adicionaVertice(self, vertice):
        if vertice not in self.lista_vertices:
            self.lista_vertices.append(vertice)
            if self.representacao == 'matriz':
                for linha in self.matriz:
                    linha.append(0)
                self.matriz.append([0] * len(self.lista_vertices))
            else:
                self.dict_vizinhos[vertice] = []
        else:
            print("Erro: Vertice ja existe no grafo.")

    def ehConectado(self):
        visitados = set()
        def dfs(vertice):
            visitados.add(vertice)
            for vizinho in self.pegaVizinhos(vertice):
                if vizinho not in visitados:
                    dfs(vizinho)
        
        if self.listaVertices:
            dfs(self.listaVertices[0])
        
        return len(visitados) == len(self.listaVertices)

    def criaAresta(self, vertice_a, vertice_b, peso=1.0):
        if not self.ponderado:
            peso = 1.0
        if vertice_a in self.lista_vertices and vertice_b in self.lista_vertices:
            if self.representacao == 'matriz':
                index_a = self.lista_vertices.index(vertice_a)
                index_b = self.lista_vertices.index(vertice_b)
                self.matriz[index_a][index_b] = peso
                if not self.direcionado:
                    self.matriz[index_b][index_a] = peso
            else:
                self.dict_vizinhos[vertice_a].append((vertice_b, peso))
                if not self.direcionado:
                    self.dict_vizinhos[vertice_b].append((vertice_a, peso))
        else:
            print("Erro: Um dos vertices fornecidos nao existe no grafo.")

    def removeVertice(self, vertice):
        if vertice in self.lista_vertices:
            index_vertice = self.lista_vertices.index(vertice)
            self.lista_vertices.pop(index_vertice)
            if self.representacao == 'matriz':
                for linha in self.matriz:
                    linha.pop(index_vertice)
                self.matriz.pop(index_vertice)
            else:
                del self.dict_vizinhos[vertice]
                for vizinhos in self.dict_vizinhos.values():
                    vizinhos[:] = [(v, p) for v, p in vizinhos if v != vertice]
        else:
            print("Erro: Vertice nao encontrado no grafo.")

    def removeAresta(self, vertice_a, vertice_b):
        if vertice_a in self.lista_vertices and vertice_b in self.lista_vertices:
            if self.representacao == 'matriz':
                index_a = self.lista_vertices.index(vertice_a)
                index_b = self.lista_vertices.index(vertice_b)
                self.matriz[index_a][index_b] = 0
                if not self.direcionado:
                    self.matriz[index_b][index_a] = 0
            else:
                self.dict_vizinhos[vertice_a] = [(v, peso) for v, peso in self.dict_vizinhos[vertice_a] if v != vertice_b]
                if not self.direcionado:
                    self.dict_vizinhos[vertice_b] = [(v, peso) for v, peso in self.dict_vizinhos[vertice_b] if v != vertice_a]
        else:
            print("Erro: Um dos vertices fornecidos nao existe no grafo.")

    def existeAresta(self, vertice_a, vertice_b):
        if vertice_a not in self.lista_vertices or vertice_b not in self.lista_vertices:
            print("Erro: Um dos vertices fornecidos nao existe no grafo.")
            return False

        if self.representacao == 'matriz':
            index_a = self.lista_vertices.index(vertice_a)
            index_b = self.lista_vertices.index(vertice_b)
            return self.matriz[index_a][index_b] != 0
        else:
            return any(v == vertice_b for v, _ in self.dict_vizinhos[vertice_a])

    def calculaIndegree(self, vertice):
        if vertice not in self.lista_vertices:
            print("Erro: O vertice fornecido nao existe no grafo.")
            return None

        indegree = 0

        if self.representacao == 'matriz':
            index_vertice = self.lista_vertices.index(vertice)
            for linha in self.matriz:
                indegree += linha[index_vertice]
        else:
            for vizinhos in self.dict_vizinhos.values():
                for v, _ in vizinhos:
                    if v == vertice:
                        indegree += 1

        return indegree
    
    def calculaOutdegree(self, vertice):
        if vertice not in self.lista_vertices:
            print("Erro: O vertice fornecido nao existe no grafo.")
            return None

        if self.representacao == 'matriz':
            index_vertice = self.lista_vertices.index(vertice)
            return sum(self.matriz[index_vertice])
        else:
            return len(self.dict_vizinhos.get(vertice, []))

    def calculaDegree(self, vertice):
        if vertice not in self.lista_vertices:
            print("Erro: O vertice fornecido nao existe no grafo.")
            return None

        if self.direcionado:
            return self.calculaIndegree(vertice) + self.calculaOutdegree(vertice)
        else:
            return self.calculaOutdegree(vertice)

    def definirPesoAresta(self, vertice_a, vertice_b, peso=None):
        if vertice_a not in self.lista_vertices or vertice_b not in self.lista_vertices:
            print("Erro: Um dos vertices fornecidos nao existe no grafo.")
            return

        if self.representacao == 'matriz':
            index_a = self.lista_vertices.index(vertice_a)
            index_b = self.lista_vertices.index(vertice_b)
            if self.ponderado:
                self.matriz[index_a][index_b] = peso
                if not self.direcionado:
                    self.matriz[index_b][index_a] = peso
            else:
                self.matriz[index_a][index_b] = 1
                if not self.direcionado:
                    self.matriz[index_b][index_a] = 1
        else:
            if self.ponderado:
                for i, (v, _) in enumerate(self.dict_vizinhos[vertice_a]):
                    if v == vertice_b:
                        self.dict_vizinhos[vertice_a][i] = (v, peso)
                        break
                else:
                    self.dict_vizinhos[vertice_a].append((vertice_b, peso))
                if not self.direcionado:
                    for i, (v, _) in enumerate(self.dict_vizinhos[vertice_b]):
                        if v == vertice_a:
                            self.dict_vizinhos[vertice_b][i] = (v, peso)
                            break
                    else:
                        self.dict_vizinhos[vertice_b].append((vertice_a, peso))
            else:
                if vertice_b not in [v for v, _ in self.dict_vizinhos[vertice_a]]:
                    self.dict_vizinhos[vertice_a].append((vertice_b, 1))
                if not self.direcionado:
                    if vertice_a not in [v for v, _ in self.dict_vizinhos[vertice_b]]:
                        self.dict_vizinhos[vertice_b].append((vertice_a, 1))

    def recuperarPesoAresta(self, vertice_a, vertice_b):
        if vertice_a not in self.lista_vertices or vertice_b not in self.lista_vertices:
            print("Erro: Um dos vertices fornecidos nao existe no grafo.")
            return None

        if self.representacao == 'matriz':
            index_a = self.lista_vertices.index(vertice_a)
            index_b = self.lista_vertices.index(vertice_b)
            return self.matriz[index_a][index_b] if self.matriz[index_a][index_b] != 0 else None
        else:
            for vizinho, peso in self.dict_vizinhos[vertice_a]:
                if vizinho == vertice_b:
                    return peso
            return None

    def calculaGrauNaoDirecionado(self, vertice):
        return self.calcula_outdegree(vertice)

    def fechamentoTransitivo(self):
        if self.representacao == 'matriz':
            mat_warshall = [linha[:] for linha in self.matriz]
            for k in range(len(mat_warshall)):
                for i in range(len(mat_warshall)):
                    for j in range(len(mat_warshall)):
                        if i != j:
                            if mat_warshall[i][j] == 0 and mat_warshall[i][k] != 0 and mat_warshall[k][j] != 0:
                                mat_warshall[i][j] = mat_warshall[i][k] + mat_warshall[k][j]
                            elif mat_warshall[i][k] != 0 and mat_warshall[k][j] != 0:
                                mat_warshall[i][j] = min(mat_warshall[i][j], mat_warshall[i][k] + mat_warshall[k][j]) if mat_warshall[i][j] != 0 else mat_warshall[i][k] + mat_warshall[k][j]
            novo_grafo = Grafos(representacao='matriz', direcionado=self.direcionado, ponderado=self.ponderado)
            novo_grafo.lista_vertices = self.lista_vertices[:]
            novo_grafo.matriz = mat_warshall
            return novo_grafo
        else:
            novo_grafo = Grafos(representacao='lista', direcionado=self.direcionado, ponderado=self.ponderado)
            for vertice in self.lista_vertices:
                novo_grafo.adicionaVertice(vertice)

            arestas_a_adicionar = set()
            for vertice in self.lista_vertices:
                visitados = set()
                self.DFSTransitivo(vertice, vertice, visitados, arestas_a_adicionar)

            for vertice_a, vertice_b in arestas_a_adicionar:
                if not novo_grafo.existe_aresta(vertice_a, vertice_b):
                    peso = self.recuperar_peso_aresta(vertice_a, vertice_b)
                    if peso is not None:
                        novo_grafo.cria_aresta(vertice_a, vertice_b, peso)

            return novo_grafo


    def DFSTransitivo(self, vertice_inicial, vertice_atual, visitados, arestas_a_adicionar):
        visitados.add(vertice_atual)
        for vizinho, peso in self.dict_vizinhos.get(vertice_atual, []):
            if vizinho not in visitados:
                arestas_a_adicionar.add((vertice_inicial, vizinho))
                self.DFSTransitivo(vertice_inicial, vizinho, visitados, arestas_a_adicionar)



    def buscaLargura(self, vertice_a, vertice_b):
        inicio = time.time()
        visitados = set()
        fila = deque([vertice_a])
        ordem_visita = []

        while fila:
            vertice_atual = fila.popleft()
            if vertice_atual not in visitados:
                visitados.add(vertice_atual)
                ordem_visita.append(vertice_atual)
                if vertice_atual == vertice_b:
                    fim = time.time()
                    return ordem_visita, fim - inicio
                for vizinho in self.pegaVizinhos(vertice_atual):
                    if vizinho not in visitados:
                        fila.append(vizinho)
        fim = time.time()
        return ordem_visita, fim - inicio



    def buscaProfundidade(self, vertice_a, vertice_b):
        inicio = time.time()
        visitados = set()
        caminho_encontrado = []

        def dfs(vertice_atual):
            if vertice_atual in visitados:
                return False
            visitados.add(vertice_atual)
            caminho_encontrado.append(vertice_atual)
            if vertice_atual == vertice_b:
                return True
            for vizinho in self.pegaVizinhos(vertice_atual):
                if dfs(vizinho):
                    return True
            caminho_encontrado.pop()
            return False
        
        resultado = dfs(vertice_a)
        fim = time.time()
        return (caminho_encontrado, fim - inicio) if resultado else None
    
    def pegaVizinhos(self, vertice):
        if self.representacao == 'matriz':
            indexVertice = self.listaVertices.index(vertice)
            vizinhos = []
            return [self.listaVertices[i] for i, val in enumerate(self.matriz[indexVertice]) if val != 0]
        else:
            return [v for v, _ in self.dictVizinhos.get(vertice, [])]




    def dijkstra(self, verticeA, verticeB):
        if not self.ponderado:
            print("Erro: Dijkstra requer um grafo ponderado.")
            return [], float('inf'), 0

        inicio = time.time()
        visitados = set()
        distancias = {vertice: float('inf') for vertice in self.listaVertices}
        distancias[verticeA] = 0
        antecessores = {vertice: None for vertice in self.listaVertices}

        while visitados != set(self.listaVertices):
            naoVisitados = {vertice: distancias[vertice] for vertice in self.listaVertices if vertice not in visitados}
            if not naoVisitados:
                break

            verticeAtual = min(naoVisitados, key=naoVisitados.get)
            visitados.add(verticeAtual)

            if self.representacao == 'matriz':
                indexVerticeAtual = self.listaVertices.index(verticeAtual)
                for indexVizinho, peso in enumerate(self.matriz[indexVerticeAtual]):
                    if peso != 0 and self.listaVertices[indexVizinho] not in visitados:
                        novaDistancia = distancias[verticeAtual] + peso
                        if novaDistancia < distancias[self.listaVertices[indexVizinho]]:
                            distancias[self.listaVertices[indexVizinho]] = novaDistancia
                            antecessores[self.listaVertices[indexVizinho]] = verticeAtual
            else:
                for vizinho, peso in self.dictVizinhos[verticeAtual]:
                    if vizinho not in visitados:
                        novaDistancia = distancias[verticeAtual] + peso
                        if novaDistancia < distancias[vizinho]:
                            distancias[vizinho] = novaDistancia
                            antecessores[vizinho] = verticeAtual

        caminho = []
        verticeAtual = verticeB
        while verticeAtual is not None:
            caminho.insert(0, verticeAtual)
            verticeAtual = antecessores[verticeAtual]

        fim = time.time()
        return caminho, distancias[verticeB], fim - inicio

    def verificarEuleriano(self):
        if not self.ehConectado():
            return False
        for vertice in self.listaVertices:
            if len(self.pegaVizinhos(vertice)) % 2 != 0:
                return False
        return True

    def distribuicaoGrau(self, nomeArquivo):
        graus = [len(self.pegaVizinhos(vertice)) for vertice in self.listaVertices]

        plt.hist(graus, bins=max(graus)+1, edgecolor='black')
        plt.title('Distribuicao de Grau dos Nos')
        plt.xlabel('Grau do No')
        plt.ylabel('Numero de Nos')
        plt.grid(True)
        plt.savefig(nomeArquivo)
        plt.close()

    def pegaVizinhos(self, vertice):
        vizinhos = []
        if vertice in self.listaVertices:
            if self.representacao == 'matriz':
                indexVertice = self.listaVertices.index(vertice)
                for i, peso in enumerate(self.matriz[indexVertice]):
                    if peso != 0:
                        vizinhos.append(self.listaVertices[i])
            else:
                for (vizinho, _) in self.dictVizinhos[vertice]:
                    vizinhos.append(vizinho)
        return vizinhos

    def pegaPeso(self, verticeA, verticeB):
        if self.representacao == 'matriz':
            indexA = self.listaVertices.index(verticeA)
            indexB = self.listaVertices.index(verticeB)
            return self.matriz[indexA][indexB]
        else:
            for vizinho, peso in self.dictVizinhos[verticeA]:
                if vizinho == verticeB:
                    return peso
            return float('inf')

    def extractMin(self, q, pesoAcumulados):
        verticeMenorCusto = None
        minWeight = float('inf')
        for vertice in q:
            if pesoAcumulados[vertice] < minWeight:
                minWeight = pesoAcumulados[vertice]
                verticeMenorCusto = vertice
        return verticeMenorCusto

    def ehConectado(self):
        if not self.listaVertices:
            return False

        visitados = set()
        pilha = [self.listaVertices[0]]

        while pilha:
            vertice = pilha.pop()
            if vertice not in visitados:
                visitados.add(vertice)
                for vizinho in self.pegaVizinhos(vertice):
                    if vizinho not in visitados:
                        pilha.append(vizinho)

        return len(visitados) == len(self.listaVertices)


    def prim(self):
        if not self.ehConectado():
            print("O grafo nao e conectado. Nao e possivel gerar a arvore geradora minima.")
            return None, 0

        predecessores = {}
        pesos = {}
        for vertice in self.listaVertices:
            predecessores[vertice] = None
            pesos[vertice] = float('inf')

        q = set(self.listaVertices)

        verticeInicial = next(iter(q))
        pesos[verticeInicial] = 0

        while q:
            u = min(q, key=lambda vertice: pesos[vertice])

            if u is None:
                break

            q.remove(u)

            for vizinho in self.pegaVizinhos(u):
                peso = self.pegaPeso(u, vizinho)
                if vizinho in q and peso < pesos[vizinho]:
                    predecessores[vizinho] = u
                    pesos[vizinho] = peso

        gPrim = Grafos(representacao=self.representacao, direcionado=False, ponderado=True)

        for vertice in self.listaVertices:
            gPrim.adicionaVertice(vertice)

        custoAcumulado = 0
        for verticeInicio, verticeFinal in predecessores.items():
            if verticeFinal is not None:
                peso = pesos[verticeInicio]
                gPrim.criaAresta(verticeInicio, verticeFinal, peso)
                custoAcumulado += peso

        return gPrim, custoAcumulado

    def pegaVizinhosPajek(self, vertice):
        vizinhos = []  
        if vertice in self.listaVertices:  
            if self.representacao == 'matriz':  
                indexVertice = self.listaVertices.index(vertice)
                for i in range(len(self.listaVertices)):
                    peso = self.matriz[indexVertice][i]
                    if peso != 0:  
                        if self.ponderado:  
                            vizinhos.append((self.listaVertices[i], peso))
                        else:  
                            vizinhos.append(self.listaVertices[i])
            else:
                for (vizinho, peso) in self.dictVizinhos[vertice]:
                    if self.ponderado:  
                        vizinhos.append((vizinho, peso))
                    else:  
                        vizinhos.append(vizinho)
        return vizinhos

    def persistePajek(self, nomeArquivo):
        with open(nomeArquivo, 'w') as f:
            f.write(f'% directed={self.direcionado}\n')
            f.write(f'% weighted={self.ponderado}\n')
            f.write(f'% representation={self.representacao}\n')
            
            f.write('*Vertices {}\n'.format(len(self.listaVertices)))
            for i, vertice in enumerate(self.listaVertices):
                f.write('{} "{}"\n'.format(i+1, vertice))
            
            if self.direcionado:
                f.write('*Arcs\n')
            else:
                f.write('*Edges\n')

            if self.representacao == 'matriz':
                for i, verticeA in enumerate(self.listaVertices):
                    for j, peso in enumerate(self.matriz[i]):
                        if peso != 0:
                            if self.ponderado:
                                f.write('{} {} {}\n'.format(i+1, j+1, peso))
                            else:
                                f.write('{} {}\n'.format(i+1, j+1))
            else:  # lista de adjacência
                edgesWritten = set()
                for vertice in self.listaVertices:
                    for vizinho, peso in self.dictVizinhos[vertice]:
                        i = self.listaVertices.index(vertice) + 1
                        j = self.listaVertices.index(vizinho) + 1
                        if (i, j) not in edgesWritten and (j, i) not in edgesWritten:
                            if self.ponderado:
                                f.write('{} {} {}\n'.format(i, j, peso))
                            else:
                                f.write('{} {}\n'.format(i, j))
                            edgesWritten.add((i, j))
                            if not self.direcionado:
                                edgesWritten.add((j, i))

    def carregaPajek(self, arquivoEntrada):
        with open(arquivoEntrada, 'r') as f:
            linhas = f.readlines()

        vertices = {}
        arestas = set()
        lendoVertices = False
        lendoArestas = False

        for linha in linhas:
            if linha.startswith('%'):
                if 'directed' in linha:
                    self.direcionado = 'true' in linha.split('=')[1].strip().lower()
                if 'weighted' in linha:
                    self.ponderado = 'true' in linha.split('=')[1].strip().lower()
                if 'representation' in linha:
                    self.representacao = linha.split('=')[1].strip().lower()
            elif linha.startswith('*Vertices'):
                lendoVertices = True
                lendoArestas = False
            elif linha.startswith('*Arcs') or linha.startswith('*Edges'):
                lendoVertices = False
                lendoArestas = True
            elif lendoVertices:
                partes = linha.strip().split(' ', 1)
                indice = int(partes[0]) - 1
                nomeVertice = partes[1].strip('"')
                vertices[indice] = nomeVertice
            elif lendoArestas:
                partes = linha.strip().split()
                if len(partes) == 3:
                    indexVerticeA = int(partes[0]) - 1
                    indexVerticeB = int(partes[1]) - 1
                    peso = float(partes[2])
                else:
                    indexVerticeA = int(partes[0]) - 1
                    indexVerticeB = int(partes[1]) - 1
                    peso = 1.0
                arestas.add((indexVerticeA, indexVerticeB, peso))

        self.listaVertices = []
        self.dictVizinhos = {}

        for indice, nomeVertice in vertices.items():
            self.adicionaVertice(nomeVertice)

        for indexVerticeA, indexVerticeB, peso in arestas:
            verticeA = vertices[indexVerticeA]
            verticeB = vertices[indexVerticeB]
            self.criaAresta(verticeA, verticeB, peso)

    def extrairComponentesConectados(self):
        if self.direcionado:
            print("Erro: Extração de componentes conectados só é aplicável a grafos não direcionados.")
            return None

        visitados = set()
        componentes = []

        def dfs(vertice, componente):
            visitados.add(vertice)
            componente.append(vertice)
            for vizinho in self.pegaVizinhos(vertice):
                if vizinho not in visitados:
                    dfs(vizinho, componente)

        for vertice in self.listaVertices:
            if vertice not in visitados:
                componente = []
                dfs(vertice, componente)
                componentes.append(componente)

        return componentes

    def extrairComponentesFortementeConectados(self):
        if not self.direcionado:
            print("Erro: Extração de componentes fortemente conectadas só é aplicável a grafos direcionados.")
            return None

        def dfs(self, vertice):
            visitados = set()
            pilha = []
            stack = [vertice]

            while stack:
                v = stack.pop()
                if v not in visitados:
                    visitados.add(v)
                    pilha.append(v)
                    for vizinho in self.pegaVizinhos(v):
                        if vizinho not in visitados:
                            stack.append(vizinho)

            return pilha


        def dfsTransposto(self, vertice, grafoTransposto):
            visitados = set()
            componente = []
            stack = [vertice]

            while stack:
                v = stack.pop()
                if v not in visitados:
                    visitados.add(v)
                    componente.append(v)
                    for vizinho in grafoTransposto.pegaVizinhos(v):
                        if vizinho not in visitados:
                            stack.append(vizinho)

            return componente

        # Passo 1: DFS no grafo original e registrar a ordem de término
        visitados = set()
        pilha = []
        for vertice in self.listaVertices:
            if vertice not in visitados:
                dfs(vertice, visitados, pilha)

        # Passo 2: Transpor o grafo
        grafoTransposto = self.transporGrafo()

        # Passo 3: DFS no grafo transposto na ordem inversa de término
        visitados.clear()
        componentesFortementeConectadas = []
        while pilha:
            vertice = pilha.pop()
            if vertice not in visitados:
                componente = []
                dfsTransposto(vertice, visitados, componente)
                componentesFortementeConectadas.append(componente)

        return componentesFortementeConectadas

    def pegaVizinhosTransposto(self, vertice):
        vizinhos = []
        if self.representacao == 'matriz':
            indexVertice = self.listaVertices.index(vertice)
            for i, linha in enumerate(self.matriz):
                if linha[indexVertice] != 0:
                    vizinhos.append(self.listaVertices[i])
        else:
            vizinhos = [v for v, _ in self.dictVizinhosTransposto.get(vertice, [])]
        return vizinhos


    def transporGrafo(self):
        grafoTransposto = Grafos(self.representacao, self.direcionado, self.ponderado)
        grafoTransposto.listaVertices = self.listaVertices[:]
        if self.representacao == 'matriz':
            grafoTransposto.matriz = [[self.matriz[j][i] for j in range(len(self.matriz))]
                                    for i in range(len(self.matriz))]
        else:
            grafoTransposto.dictVizinhos = {vertice: [] for vertice in self.listaVertices}
            for vertice, vizinhos in self.dictVizinhos.items():
                for vizinho, peso in vizinhos:
                    grafoTransposto.dictVizinhos[vizinho].append((vertice, peso))
        return grafoTransposto

    def calcularCentralidadeDeGrau(self):
        centralidade = {}
        for vertice in self.listaVertices:
            if self.direcionado:
                centralidade[vertice] = {
                    'inDegree': self.calculaIndegree(vertice),
                    'outDegree': self.calculaOutdegree(vertice),
                    'total': self.calculaIndegree(vertice) + self.calculaOutdegree(vertice)
                }
            else:
                centralidade[vertice] = self.calculaDegree(vertice)
        return centralidade
    
    def calcularCentralidadeDeIntermediacao(self):
        centralidade = {v: 0.0 for v in self.listaVertices}
        
        for s in self.listaVertices:
            # Etapa 1: Realizar uma BFS a partir de s
            S = []
            P = {v: [] for v in self.listaVertices}
            sigma = {v: 0 for v in self.listaVertices}
            d = {v: -1 for v in self.listaVertices}
            sigma[s] = 1
            d[s] = 0
            Q = deque([s])
            
            while Q:
                v = Q.popleft()
                S.append(v)
                for w in self.pegaVizinhos(v):
                    if d[w] < 0:
                        Q.append(w)
                        d[w] = d[v] + 1
                    if d[w] == d[v] + 1:
                        sigma[w] += sigma[v]
                        P[w].append(v)
            
            # Etapa 2: Acumular dependências
            delta = {v: 0.0 for v in self.listaVertices}
            while S:
                w = S.pop()
                for v in P[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != s:
                    centralidade[w] += delta[w]
        
        # Normalizar para grafos não direcionados
        if not self.direcionado:
            for v in centralidade:
                centralidade[v] /= 2.0
        
        return centralidade

    def calcularCentralidadeDeProximidade(self):
        proximidade = {vertice: 0 for vertice in self.listaVertices}

        for vertice in self.listaVertices:
            distancias = {v: float('inf') for v in self.listaVertices}
            distancias[vertice] = 0
            Q = deque([vertice])
            
            while Q:
                u = Q.popleft()
                for vizinho in self.pegaVizinhos(u):
                    if distancias[vizinho] == float('inf'):
                        Q.append(vizinho)
                        distancias[vizinho] = distancias[u] + 1
            
            somaDistancias = sum(d for d in distancias.values() if d != float('inf'))
            if somaDistancias > 0:
                proximidade[vertice] = (len(self.listaVertices) - 1) / somaDistancias

        return proximidade


    def calcularExcentricidade(self):
        if not self.ehConectado():
            print("Erro: O grafo não é conexo. Não é possível calcular excentricidade.")
            return {v: None for v in self.listaVertices}

        excentricidade = {}
        for vertice in self.listaVertices:
            distancias = self.calcularDistancias(vertice)
            excentricidade[vertice] = max(distancias.values())

        return excentricidade




    def calcularDistancias(self, verticeInicial):
        distancias = {v: float('inf') for v in self.listaVertices}
        distancias[verticeInicial] = 0
        Q = deque([verticeInicial])
        
        while Q:
            verticeAtual = Q.popleft()
            for vizinho in self.pegaVizinhos(verticeAtual):
                if distancias[vizinho] == float('inf'):
                    distancias[vizinho] = distancias[verticeAtual] + 1
                    Q.append(vizinho)
        
        return distancias

    def calcularDiametro(self):
        if not self.ehConectado():
            print("Erro: O grafo não é conexo. Não é possível calcular o diâmetro.")
            return None
        
        excentricidades = self.calcularExcentricidade()
        diametro = max(excentricidades.values())
        
        return diametro

    def calcularRaio(self):
        if not self.ehConectado():
            print("Erro: O grafo não é conexo. Não é possível calcular o raio.")
            return None
        
        excentricidades = self.calcularExcentricidade()
        raio = min(excentricidades.values())
        
        return raio

    def pegaVizinhos(self, vertice):
        vizinhos = []
        if vertice in self.listaVertices:
            if self.representacao == 'matriz':
                indexVertice = self.listaVertices.index(vertice)
                for i, peso in enumerate(self.matriz[indexVertice]):
                    if peso != 0:
                        vizinhos.append(self.listaVertices[i])
            else:
                for vizinho, peso in self.dictVizinhos[vertice]:
                    vizinhos.append(vizinho)
        return vizinhos

    def removerAresta(self, verticeA, verticeB):
        if verticeA in self.listaVertices and verticeB in self.listaVertices:
            if self.representacao == 'matriz':
                indexA = self.listaVertices.index(verticeA)
                indexB = self.listaVertices.index(verticeB)
                self.matriz[indexA][indexB] = 0
                if not self.direcionado:
                    self.matriz[indexB][indexA] = 0
            else:
                self.dictVizinhos[verticeA] = [(v, p) for v, p in self.dictVizinhos[verticeA] if v != verticeB]
                if not self.direcionado:
                    self.dictVizinhos[verticeB] = [(v, p) for v, p in self.dictVizinhos[verticeB] if v != verticeA]
        else:
            print("Erro: Um dos vertices fornecidos nao existe no grafo.")

    def detectarComunidadesGirvanNewman(self, numComunidades):
        def removerArestasDeMaiorCentralidade(grafo):
            edgeCentralities = grafo.calcularCentralidadeDeIntermediacaoAresta()
            maxCentralidade = max(edgeCentralities.values())
            arestasParaRemover = [aresta for aresta, centralidade in edgeCentralities.items() if centralidade == maxCentralidade]
            if arestasParaRemover:
                aresta = arestasParaRemover[0]
                verticeA, verticeB = tuple(aresta)
                grafo.removerAresta(verticeA, verticeB)
                print(f"Aresta removida: {verticeA} - {verticeB}")
            return grafo

        def dfs(vertice, componenteAtual, visitados):
            visitados.add(vertice)
            componenteAtual.append(vertice)
            for vizinho in grafoCopia.pegaVizinhos(vertice):
                if vizinho not in visitados:
                    dfs(vizinho, componenteAtual, visitados)

        def calcularComponentesConectados():
            componentes = []
            visitados = set()
            for vertice in grafoCopia.listaVertices:
                if vertice not in visitados:
                    componenteAtual = []
                    dfs(vertice, componenteAtual, visitados)
                    componentes.append(componenteAtual)
            return componentes

        grafoCopia = self.copiaGrafo()
        comunidades = calcularComponentesConectados()

        while len(comunidades) < numComunidades:
            print(f"Removendo arestas para formar mais comunidades...")
            grafoCopia = removerArestasDeMaiorCentralidade(grafoCopia)
            novasComunidades = calcularComponentesConectados()
            print(f"Comunidades formadas: {novasComunidades}")

            # Verificar se houve mudança no número de comunidades
            if len(novasComunidades) == len(comunidades):
                break

            comunidades = novasComunidades

        return comunidades
# ---------------------------------------------------------------------------------------------------------------------------------

def copiaGrafo(self):
    novoGrafo = Grafos(representacao=self.representacao, direcionado=self.direcionado, ponderado=self.ponderado)
    novoGrafo.listaVertices = self.listaVertices[:]
    if self.representacao == 'matriz':
        novoGrafo.matriz = [linha[:] for linha in self.matriz]
    else:
        novoGrafo.dictVizinhos = {vertice: vizinhos[:] for vertice, vizinhos in self.dictVizinhos.items()}
    return novoGrafo

def calcularCentralidadeDeIntermediacaoAresta(self):
    edgeCentrality = {frozenset((v, u)): 0 for v in self.listaVertices for u in self.pegaVizinhos(v)}
    for s in self.listaVertices:
        pred = {w: [] for w in self.listaVertices}
        sigma = dict.fromkeys(self.listaVertices, 0)
        sigma[s] = 1
        dist = dict.fromkeys(self.listaVertices, -1)
        dist[s] = 0
        Q = deque([s])
        S = []
        while Q:
            v = Q.popleft()
            S.append(v)
            for w in self.pegaVizinhos(v):
                if dist[w] < 0:
                    Q.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = dict.fromkeys(self.listaVertices, 0)
        while S:
            w = S.pop()
            for v in pred[w]:
                c = (sigma[v] / sigma[w]) * (1 + delta[w])
                edge = frozenset((v, w))
                edgeCentrality[edge] += c
                delta[v] += c
    return edgeCentrality

import heapq
from collections import defaultdict, deque

def carregaEProcessaCSV(nomeArquivo):
    import pandas as pd
    import ast
    from collections import defaultdict

    df = pd.read_csv(nomeArquivo)

    # Inicializar o grafo
    grafo = defaultdict(lambda: defaultdict(int))

    # Iterar sobre as linhas do DataFrame
    for index, row in df.iterrows():
        nomesAutores = ast.literal_eval(row['NOMES_AJUSTADOS'])
        # Verificar se há mais de um autor na publicação
        if len(nomesAutores) < 2:
            continue  # Ignorar publicações com apenas um autor listado
        # Atualizar o grafo com as conexões entre os autores
        for i in range(len(nomesAutores)):
            for j in range(i + 1, len(nomesAutores)):
                autor1, autor2 = nomesAutores[i], nomesAutores[j]
                # Incrementar a contagem de publicações entre os autores
                grafo[autor1][autor2] += 1
                grafo[autor2][autor1] += 1

    # Filtrar autores sem colaborações significativas
    grafoFiltrado = {autor: {vizinho: peso for vizinho, peso in vizinhos.items() if peso > 0} for autor, vizinhos in grafo.items()}
    grafoFiltrado = {autor: vizinhos for autor, vizinhos in grafoFiltrado.items() if len(vizinhos) > 0}

    return grafoFiltrado

# Quais pares de autores são os mais produtivos dentro da rede? Elenque os 10 pares de autores
# mais produtivos da rede.
def encontrarParesMaisProdutivos(grafo, n=10):
    # Criar uma lista de pares (aresta, peso)
    paresProdutivos = []
    for autor, vizinhos in grafo.items():
        for vizinho, peso in vizinhos.items():
            if (vizinho, autor, peso) not in paresProdutivos:  # Evitar duplicação de pares
                paresProdutivos.append((autor, vizinho, peso))
    
    # Ordenar a lista de pares pelo peso em ordem decrescente
    paresProdutivos.sort(key=lambda x: x[2], reverse=True)
    
    # Selecionar os top n pares
    return paresProdutivos[:n]

# Quantas componentes o grafo possui?
def contarComponentesConectados(grafo):
    def dfs(vertice, visitados):
        stack = [vertice]
        while stack:
            v = stack.pop()
            if v not in visitados:
                visitados.add(v)
                stack.extend(set(grafo[v].keys()) - visitados)

    visitados = set()
    componentes = 0

    for vertice in grafo:
        if vertice not in visitados:
            dfs(vertice, visitados)
            componentes += 1

    return componentes

# Calcular a distribuição dos graus dos nós da rede
def calcularDistribuicaoDeGrau(grafo):
    graus = [len(vizinhos) for vizinhos in grafo.values()]
    return graus

def encontrarAutoresMaisInfluentes(grafo, n=10):
    centralidadeGrau = {autor: len(vizinhos) for autor, vizinhos in grafo.items()}
    autoresMaisInfluentes = sorted(centralidadeGrau.items(), key=lambda x: x[1], reverse=True)[:n]
    return autoresMaisInfluentes

def calcularCentralidadeProximidade(grafo):
    proximidade = {vertice: 0 for vertice in grafo}

    for vertice in grafo:
        distancias = {v: float('inf') for v in grafo}
        distancias[vertice] = 0
        pq = [(0, vertice)]
        
        while pq:
            distanciaAtual, u = heapq.heappop(pq)
            if distanciaAtual > distancias[u]:
                continue
            
            for vizinho, peso in grafo[u].items():
                distancia = distanciaAtual + 1 / peso
                if distancia < distancias[vizinho]:
                    distancias[vizinho] = distancia
                    heapq.heappush(pq, (distancia, vizinho))
        
        somaDistancias = sum(d for d in distancias.values() if d != float('inf'))
        if somaDistancias > 0:
            proximidade[vertice] = (len(grafo) - 1) / somaDistancias

    return proximidade

def encontrarAutoresMaisInfluentesProximidade(grafo, n=10):
    proximidade = calcularCentralidadeProximidade(grafo)
    proximidadeOrdenada = sorted(proximidade.items(), key=lambda x: x[1], reverse=True)
    return proximidadeOrdenada[:n]

def encontraComponentesConectados(grafo):
    visitados = set()
    componentes = []

    def dfs(v, componente):
        visitados.add(v)
        componente.append(v)
        for vizinho in grafo[v]:
            if vizinho not in visitados:
                dfs(vizinho, componente)

    for vertice in grafo:
        if vertice not in visitados:
            componente = []
            dfs(vertice, componente)
            componentes.append(componente)

    return componentes

def calculaDistancias(grafo, verticeInicial):
    distancias = {v: float('inf') for v in grafo}
    distancias[verticeInicial] = 0
    Q = deque([verticeInicial])

    while Q:
        verticeAtual = Q.popleft()
        for vizinho in grafo[verticeAtual]:
            if distancias[vizinho] == float('inf'):
                distancias[vizinho] = distancias[verticeAtual] + 1
                Q.append(vizinho)

    return distancias

def calcularExcentricidade(grafo):
    componentes = encontraComponentesConectados(grafo)
    excentricidade = {}
    for componente in componentes:
        for vertice in componente:
            distancias = calculaDistancias(grafo, vertice)
            maxDistancia = max(distancias[v] for v in componente)
            excentricidade[vertice] = maxDistancia
    return excentricidade

def calcularComponentesConectados(grafo):
    visitados = set()
    componentes = []

    def dfs(vertice, componenteAtual):
        visitados.add(vertice)
        componenteAtual.append(vertice)
        for vizinho in grafo[vertice]:
            if vizinho not in visitados:
                dfs(vizinho, componenteAtual)

    for vertice in grafo:
        if vertice not in visitados:
            componenteAtual = []
            dfs(vertice, componenteAtual)
            componentes.append(componenteAtual)
    
    return componentes

def calcularCentralidadeExcentricidade(grafo):
    def bfs(verticeInicial):
        distancias = {v: float('inf') for v in grafo}
        distancias[verticeInicial] = 0
        fila = deque([verticeInicial])
        
        while fila:
            verticeAtual = fila.popleft()
            for vizinho in grafo[verticeAtual]:
                if distancias[vizinho] == float('inf'):
                    distancias[vizinho] = distancias[verticeAtual] + 1
                    fila.append(vizinho)
        
        return distancias
    
    excentricidade = {}
    componentes = calcularComponentesConectados(grafo)
    
    for componente in componentes:
        for vertice in componente:
            distancias = bfs(vertice)
            # Filtrar apenas as distâncias finitas (dentro do mesmo componente)
            distanciasFinitas = [dist for dist in distancias.values() if dist < float('inf')]
            if distanciasFinitas:
                excentricidade[vertice] = max(distanciasFinitas)
            else:
                excentricidade[vertice] = float('inf')
    
    return excentricidade

def encontrarAutoresMaisInfluentesExcentricidade(grafo, n=10):
    excentricidade = calcularCentralidadeExcentricidade(grafo)
    # Ordenar pela menor excentricidade (mais influente)
    autoresOrdenados = sorted(excentricidade.items(), key=lambda x: x[1])
    return autoresOrdenados[:n]

def calcularDiametro(grafo):
    def bfs(v):
        dist = {v: 0}
        queue = [v]
        for u in queue:
            for neighbor in grafo[u]:
                if neighbor not in dist:
                    dist[neighbor] = dist[u] + 1
                    queue.append(neighbor)
        return dist

    maxDist = 0
    for v in grafo:
        dist = bfs(v)
        maxDist = max(maxDist, max(dist.values()))

    return maxDist

def calcularRaio(grafo):
    def bfs(v):
        dist = {v: 0}
        queue = [v]
        for u in queue:
            for neighbor in grafo[u]:
                if neighbor not in dist:
                    dist[neighbor] = dist[u] + 1
                    queue.append(neighbor)
        return dist

    minMaxDist = float('inf')
    for v in grafo:
        dist = bfs(v)
        minMaxDist = min(minMaxDist, max(dist.values()))

    return minMaxDist

def calcularCentralidadeIntermediacao(grafo):
    centralidade = defaultdict(int)
    
    for s in grafo:
        stack = []
        pred = {v: [] for v in grafo}
        sigma = dict.fromkeys(grafo, 0)
        sigma[s] = 1
        dist = dict.fromkeys(grafo, -1)
        dist[s] = 0
        Q = deque([s])
        while Q:
            v = Q.popleft()
            stack.append(v)
            for w in grafo[v]:
                if dist[w] < 0:
                    Q.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = dict.fromkeys(grafo, 0)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                edge = tuple(sorted((v, w)))
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if v != s:
                    centralidade[edge] += delta[w]
    return centralidade

def encontrarArestaMaisRelevante(grafo, n=10):
    centralidade = calcularCentralidadeIntermediacao(grafo)
    # Ordenar as arestas pela centralidade de intermediação em ordem decrescente
    arestasRelevantes = sorted(centralidade.items(), key=lambda x: x[1], reverse=True)
    # Selecionar as top n arestas
    return arestasRelevantes[:n]

def encontraMaiorComponenteConectada(grafo):
    visitados = set()

    def dfs(vertice):
        visitados.add(vertice)
        componente = [vertice]
        for vizinho in grafo[vertice]:
            if vizinho not in visitados:
                componente.extend(dfs(vizinho))
        return componente

    maiorComponente = []
    for vertice in grafo:
        if vertice not in visitados:
            componenteAtual = dfs(vertice)
            if len(componenteAtual) > len(maiorComponente):
                maiorComponente = componenteAtual

    return maiorComponente

def calculaMediaDistanciaGeodesicas(grafo, maiorComponente):
    from collections import deque

    def bfsDistancias(origem):
        distancias = {vertice: float('inf') for vertice in maiorComponente}
        distancias[origem] = 0
        fila = deque([origem])

        while fila:
            verticeAtual = fila.popleft()
            for vizinho in grafo[verticeAtual]:
                if distancias[vizinho] == float('inf'):
                    distancias[vizinho] = distancias[verticeAtual] + 1
                    fila.append(vizinho)

        return distancias

    somaDistancias = 0
    contador = 0

    for vertice in maiorComponente:
        distancias = bfsDistancias(vertice)
        for destino in distancias:
            if distancias[destino] != float('inf') and vertice != destino:
                somaDistancias += distancias[destino]
                contador += 1

    mediaDistancia = somaDistancias / contador if contador > 0 else 0
    return mediaDistancia

def girvanNewman(grafo, numComunidades):
    def removerArestasMaiorCentralidade(grafo):
        centralidade = calcularCentralidadeIntermediacao(grafo)
        arestaMaisRelevante = max(centralidade, key=centralidade.get)
        verticeA, verticeB = tuple(arestaMaisRelevante)
        grafo[verticeA].pop(verticeB, None)
        grafo[verticeB].pop(verticeA, None)

    def calcularComponentesConectados(grafo):
        visitados = set()
        componentes = []

        def dfs(vertice, componente):
            visitados.add(vertice)
            componente.append(vertice)
            for vizinho in grafo[vertice]:
                if vizinho not in visitados:
                    dfs(vizinho, componente)

        for vertice in grafo:
            if vertice not in visitados:
                componente = []
                dfs(vertice, componente)
                componentes.append(componente)

        return componentes

    comunidades = calcularComponentesConectados(grafo)
    while len(comunidades) < numComunidades:
        removerArestasMaiorCentralidade(grafo)
        comunidades = calcularComponentesConectados(grafo)

    return comunidades

def identificarAutoresSignificativos(comunidade, grafo):
    # Usar centralidade de grau e centralidade de proximidade como métricas
    centralidadeGrau = {autor: len(grafo[autor]) for autor in comunidade}
    proximidade = calcularCentralidadeProximidade({autor: grafo[autor] for autor in comunidade})
    
    # Combinar métricas
    significancia = {autor: (centralidadeGrau[autor] + proximidade[autor]) for autor in comunidade}

    # Ordenar por significância
    autoresSignificativos = sorted(significancia.items(), key=lambda x: x[1], reverse=True)
    
    return autoresSignificativos

if __name__ == '__main__':
    grafo = carregaEProcessaCSV('tabela_artigos_limpa.csv')

    # Questão 1
    paresMaisProdutivos = encontrarParesMaisProdutivos(grafo)

    # Encontrar a maior componente conectada
    maiorComponente = encontraMaiorComponenteConectada(grafo)

    # Criar um subgrafo da maior componente conectada
    subgrafo = {autor: grafo[autor] for autor in maiorComponente}

    # Aplicar Girvan-Newman para encontrar 4 comunidades
    comunidades = girvanNewman(subgrafo, 4)


    componentes = contarComponentesConectados(grafo)
    DistribuicaoDeGrau = calcularDistribuicaoDeGrau(grafo)
    autoresMaisInfluentesGrau = encontrarAutoresMaisInfluentes(grafo)
    autoresInfluentesExcentricidade = encontrarAutoresMaisInfluentes(grafo)
    autoresMaisInfluentesProximidade = encontrarAutoresMaisInfluentesProximidade(grafo)
    autoresInfluentesExcentricidade = encontrarAutoresMaisInfluentesExcentricidade(grafo)

    diametro = calcularDiametro(grafo)
    raio = calcularRaio(grafo)

    arestaMaisRelevante = encontrarArestaMaisRelevante(grafo)

    # Calcular a média das distâncias geodésicas da maior componente
    mediaDistancia = calculaMediaDistanciaGeodesicas(grafo, maiorComponente)
    print(f"\nQUESTÃO 1)\nPares mais produtivos: {paresMaisProdutivos}")
    print(f"\nQUESTÃO 2)\nNúmero de componentes no grafo: {componentes}")
    print(f"\nQUESTÃO 3)\nDistribuição dos graus dos nós na rede: {DistribuicaoDeGrau}")
    print(f"\nQUESTÃO 4)\n10 autores mais influentes perante a métrica de centralidade de grau: {autoresMaisInfluentesGrau}")
    # print(f"\nQUESTÃO 5)\n10 autores mais influentes perante a métrica de centralidade de intermediação: {autoresInfluentesExcentricidade}")
    print(f"\nQUESTÃO 6)\n10 autores mais influentes perante a métrica de centralidade de proximidade: {autoresInfluentesExcentricidade}")
    print(f"\nQUESTÃO 7)\n10 autores mais influentes perante a métrica de centralidade de excentricidade: {autoresInfluentesExcentricidade}")
    print(f"\nQUESTÃO 8)\nDiâmetro do grafo: {diametro} Raio do grafo: {raio}")
    print(f"\nQUESTÃO 9)\nArestas mais relevantes c/ centralidade de intermediação: {arestaMaisRelevante}")
    print(f"\nQUESTÃO 10)\nMédia das distâncias geodésicas da maior componente do grafo: {mediaDistancia}")
    print(F"\nQUESTÃO 11)\n")
    # Identificar e discutir os autores mais significativos em cada comunidade
    for i, comunidade in enumerate(comunidades):
        print(f"\nComunidade {i + 1}:")
        autoresSignificativos = identificarAutoresSignificativos(comunidade, grafo)
        for autor, significancia in autoresSignificativos:
            print(f"Autor: {autor}, Significância: {significancia}")
