/**
 * Hands on 4 - Transformaciones Matematicas en 3D
 * Arturo Morales Pedroza y Christian Oswaldo Luevano Zaragoza
 */

#include "raylib.h"
#include <cmath>

int main()
{
    InitWindow(1300, 800, "Practica 5 - Transformaciones Matematicas en 3D");

    // ── Camara orbital ──────────────────────────────────────────────────────
    Camera3D camera = { 0 };
    camera.position   = { 0.0f, 22.0f, 22.0f };
    camera.target     = { 0.0f,  0.0f,  2.0f };
    camera.up         = { 0.0f,  1.0f,  0.0f };
    camera.fovy       = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    SetTargetFPS(60);

    // ── Estado: Rebote ──────────────────────────────────────────────────────
    // y_{n+1} = y_n + vy * dt   |   vy se invierte al tocar el limite
    float reboteY  = 1.0f;
    float velRebY  = 2.8f;   // px/s (unidades de mundo)

    // ── Estado: Traslacion ──────────────────────────────────────────────────
    // x_{n+1} = x_n + vx * dt
    float traslX   = -2.0f;
    float velTraslX = 1.8f;

    while (!WindowShouldClose())
    {
        float dt = GetFrameTime();
        UpdateCamera(&camera, CAMERA_ORBITAL);
        float t = (float)GetTime();

        // ── 1. Traslacion ────────────────────────────────────────────────
        traslX += velTraslX * dt;
        if (traslX >  2.0f || traslX < -2.0f) velTraslX *= -1.0f;

        // ── 2. Rotacion ──────────────────────────────────────────────────
        // x' = cx + r*cos(omega*t),   z' = cz + r*sin(omega*t)
        float angRot = t * 2.2f;
        float rotX   = cosf(angRot) * 1.6f;
        float rotZ   = sinf(angRot) * 1.6f;

        // ── 3. Rebote ────────────────────────────────────────────────────
        // vy += g * dt  ->  aqui simplificado con vel fija y rebote en limites
        reboteY += velRebY * dt;
        if (reboteY > 4.2f || reboteY < 0.45f) velRebY *= -1.0f;

        // ── 4. Movimiento Senoidal ───────────────────────────────────────
        // y(t) = cy + A * sin(omega * t)
        float senoY = 2.0f + sinf(t * 2.5f) * 1.6f;

        // ── 5. Trayectoria Parabolica ────────────────────────────────────
        // x(t) = x0 + vx*t,  y(t) = y0 + vy*t + 0.5*g*t^2
        float frac   = fmodf(t * 0.7f, 1.0f);      // parametro 0..1
        float trayX  = -2.0f + frac * 4.0f;         // avance horizontal
        float trayY  = 0.45f + 5.5f * frac * (1.0f - frac); // parabola

        // ── 6. Orbita ────────────────────────────────────────────────────
        // xp = cx + R*cos(omega*t),   zp = cz + R*sin(omega*t)
        float Rorb   = 2.0f;
        float orbX   = cosf(t * 1.1f) * Rorb;
        float orbZ   = sinf(t * 1.1f) * Rorb;

        // ── Plataformas: cuadricula 3x2 ─────────────────────────────────
        // Fila 1: z = -4   posiciones x = -8,  0,  8
        // Fila 2: z =  4   posiciones x = -8,  0,  8
        float ROW1 = -4.0f, ROW2 = 4.0f;
        float C1 = -8.0f,   C2 = 0.0f, C3 = 8.0f;

        BeginDrawing();
        ClearBackground(RAYWHITE);

        BeginMode3D(camera);

            DrawGrid(36, 1.0f);

            // ── Plataformas (planos grises) ──────────────────────────────
            DrawCube({ C1, 0.05f, ROW1 }, 5.0f, 0.1f, 5.0f, LIGHTGRAY);
            DrawCube({ C2, 0.05f, ROW1 }, 5.0f, 0.1f, 5.0f, LIGHTGRAY);
            DrawCube({ C3, 0.05f, ROW1 }, 5.0f, 0.1f, 5.0f, LIGHTGRAY);
            DrawCube({ C1, 0.05f, ROW2 }, 5.0f, 0.1f, 5.0f, LIGHTGRAY);
            DrawCube({ C2, 0.05f, ROW2 }, 5.0f, 0.1f, 5.0f, LIGHTGRAY);
            DrawCube({ C3, 0.05f, ROW2 }, 5.0f, 0.1f, 5.0f, LIGHTGRAY);

            // ══════════════════════════════════════════════════════════════
            // 1. TRASLACION  (C1, ROW1)  — x' = x + vx*dt
            // ══════════════════════════════════════════════════════════════
            // Carril de referencia
            DrawLine3D({ C1 - 2.0f, 0.45f, ROW1 },
                       { C1 + 2.0f, 0.45f, ROW1 }, SKYBLUE);
            DrawSphere({ C1 - 2.0f, 0.45f, ROW1 }, 0.1f, SKYBLUE);
            DrawSphere({ C1 + 2.0f, 0.45f, ROW1 }, 0.1f, SKYBLUE);
            // Esfera en movimiento
            DrawSphere({ C1 + traslX, 0.45f, ROW1 }, 0.45f, BLUE);

            // ══════════════════════════════════════════════════════════════
            // 2. ROTACION  (C2, ROW1)  — x'=cx+r*cos(t)  z'=cz+r*sin(t)
            // ══════════════════════════════════════════════════════════════
            // Centro fijo + orbita
            DrawSphere({ C2, 0.45f, ROW1 }, 0.2f, DARKGREEN);
            DrawSphereWires({ C2, 0.45f, ROW1 }, 1.6f, 20, 20, DARKGREEN);
            // Radio (linea)
            DrawLine3D({ C2, 0.45f, ROW1 },
                       { C2 + rotX, 0.45f, ROW1 + rotZ }, GREEN);
            // Punto rotante
            DrawSphere({ C2 + rotX, 0.45f, ROW1 + rotZ }, 0.45f, GREEN);

            // ══════════════════════════════════════════════════════════════
            // 3. REBOTE  (C3, ROW1)  — vy += g*dt;  vy*=-1 en limites
            // ══════════════════════════════════════════════════════════════
            // Plataforma de rebote
            DrawCube({ C3, 0.15f, ROW1 }, 2.0f, 0.3f, 2.0f, GRAY);
            // Linea de limite superior
            DrawLine3D({ C3, 0.3f, ROW1 }, { C3, 4.2f, ROW1 }, RED);
            DrawSphere({ C3, 4.2f, ROW1 }, 0.1f, RED);
            // Esfera rebotante
            DrawSphere({ C3, reboteY, ROW1 }, 0.45f, RED);

            // ══════════════════════════════════════════════════════════════
            // 4. SENOIDAL  (C1, ROW2)  — y = cy + A*sin(omega*t)
            // ══════════════════════════════════════════════════════════════
            // Eje vertical de referencia
            DrawLine3D({ C1, 0.0f, ROW2 }, { C1, 4.0f, ROW2 }, PURPLE);
            DrawSphere({ C1, 4.0f, ROW2 }, 0.1f, PURPLE);
            DrawSphere({ C1, 0.45f, ROW2 }, 0.1f, PURPLE);
            // Esfera senoidal
            DrawSphere({ C1, senoY, ROW2 }, 0.45f, VIOLET);

            // ══════════════════════════════════════════════════════════════
            // 5. TRAYECTORIA  (C2, ROW2)  — parabolica x+y(t)
            // ══════════════════════════════════════════════════════════════
            // Traza parabolica estatica
            for (int i = 0; i < 60; i++)
            {
                float p1 = i       / 60.0f;
                float p2 = (i + 1) / 60.0f;
                float x1 = -2.0f + p1 * 4.0f;
                float y1 = 0.45f + 5.5f * p1 * (1.0f - p1);
                float x2 = -2.0f + p2 * 4.0f;
                float y2 = 0.45f + 5.5f * p2 * (1.0f - p2);
                DrawLine3D({ C2 + x1, y1, ROW2 },
                           { C2 + x2, y2, ROW2 }, ORANGE);
            }
            // Esfera en trayectoria
            DrawSphere({ C2 + trayX, trayY, ROW2 }, 0.45f, ORANGE);

            // ══════════════════════════════════════════════════════════════
            // 6. ORBITA  (C3, ROW2)  — xp=cx+R*cos(t)  zp=cz+R*sin(t)
            // ══════════════════════════════════════════════════════════════
            // Sol (centro)
            DrawSphere({ C3, 0.45f, ROW2 }, 0.35f, YELLOW);
            // Trayectoria orbital
            DrawSphereWires({ C3, 0.45f, ROW2 }, Rorb, 24, 24, GOLD);
            // Planeta
            DrawSphere({ C3 + orbX, 0.45f, ROW2 + orbZ }, 0.45f, MAROON);

        EndMode3D();

        // ── Labels 2D ────────────────────────────────────────────────────
        DrawText("Practica 5 - Transformaciones Matematicas en 3D", 20, 18, 26, DARKGRAY);

        // Fila superior
        DrawText("1. Traslacion",      70,  78, 20, BLUE);
        DrawText("x' = x + vx*dt",     70, 102, 16, DARKGRAY);

        DrawText("2. Rotacion",        430,  78, 20, DARKGREEN);
        DrawText("x'=cx+r*cos(t)  z'=cz+r*sin(t)", 430, 102, 14, DARKGRAY);

        DrawText("3. Rebote",          830,  78, 20, RED);
        DrawText("vy*=-1 al tocar limite",  830, 102, 16, DARKGRAY);

        // Fila inferior
        DrawText("4. Senoidal",         70, 680, 20, VIOLET);
        DrawText("y = cy + A*sin(w*t)", 70, 704, 16, DARKGRAY);

        DrawText("5. Trayectoria",     430, 680, 20, ORANGE);
        DrawText("y = y0+vy*t+0.5*g*t^2", 430, 704, 14, DARKGRAY);

        DrawText("6. Orbita",          830, 680, 20, MAROON);
        DrawText("xp=cx+R*cos(t)  zp=cz+R*sin(t)", 830, 704, 14, DARKGRAY);

        DrawText("Camara orbital automatica  |  ESC para salir", 20, 758, 18, GRAY);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}