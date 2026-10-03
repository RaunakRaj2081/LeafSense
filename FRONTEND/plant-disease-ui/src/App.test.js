import { render, screen } from "@testing-library/react";
import App from "./App";

test("renders Plant Disease Detection application", () => {
  render(<App />);

  const heading = screen.getByText(
    /Plant Disease Detection/i
  );

  expect(heading).toBeInTheDocument();
});