from l_reg import *
file = 'Housing.csv'
x_train, x_test, y_train, y_test = read_and_split_data(file)
w_init = np.zeros(x_train.shape[1])
b_init = 0
iterations = 10000
tmp_alpha = 0.0007

w_final, b_final, J_hist = gradient_descent(x_train, y_train)

features = ["area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "parking", "prefarea", "furnishingstatus"]
target = "price"
y_pred = predict(x_train, w_final, b_final)
mse, mae, r2 = calc_metrics(y_train, y_pred)
cost = compute_cost(x_train, y_train, w_final, b_final)


# print(f"\nFinal parameters:")
# print(f"w: {w_final}")
# print(f"b: {b_final:.4f}")
# print(f"Final cost: {cost:.6f}")

y_pred_train = predict(x_train, w_final, b_final)
y_pred_test = predict(x_test, w_final, b_final)

print("\nFinal metrics:")
print(f"MSE: {mse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"R²: {r2:.6f}")

# c,w = compare_predictions(x_test, y_test, w_final, b_final)
# print("correct",c)
# print("Wrong",w)

# w_final, b_final, J_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iterations)

plot_all_graphs(J_hist, y_train, y_pred_train, y_test, y_pred_test)

# print(predict())
# plt.plot(J_hist)
# plt.xlabel('Iteration')
# plt.ylabel('Cost')
# plt.title('Cost vs. Iteration')
# plt.show()